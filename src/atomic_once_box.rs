use alloc::boxed::Box;
use core::{
    fmt::{self, Debug, Formatter},
    marker::PhantomData,
    mem::{forget, transmute},
    ptr::null_mut,
    sync::atomic::{AtomicPtr, Ordering},
};

use crate::atomic_option_box::{from_ptr, into_ptr};

/// A type that holds a single `Option<Box<T>>` value which only ever gets one
/// non-`None` value (unless there is exclusive access) and can be safely shared
/// between threads.
pub struct AtomicOnceBox<T> {
    ptr: AtomicPtr<T>,

    /// This effectively makes `AtomicBox<T>` non-`Send` and non-`Sync` if `T`
    /// is non-`Send`.
    phantom: PhantomData<Box<T>>,
}

/// Mark `AtomicOnceBox<T>` as safe to share across threads.
///
/// This is safe because shared access to an `AtomicOnceBox<T>` does not
/// provide shared access to any `T` value. However, it does provide the
/// ability to get a `Box<T>` from another thread, so `T: Send` is required.
unsafe impl<T> Sync for AtomicOnceBox<T> where T: Send {}

impl<T> AtomicOnceBox<T> {
    /// Creates a new `AtomicOnceBox` with the given value.
    ///
    /// # Examples
    ///
    ///     use atomicbox::AtomicOnceBox;
    ///
    ///     let atomic_box = AtomicOnceBox::new(Some(Box::new(0)));
    ///
    pub fn new(value: Option<Box<T>>) -> AtomicOnceBox<T> {
        AtomicOnceBox {
            ptr: AtomicPtr::new(into_ptr(value)),
            phantom: PhantomData,
        }
    }

    /// Creates a new `AtomicOnceBox` with no value.
    ///
    /// Equivalent to `AtomicOnceBox::new(None)`, but can be used in `const`
    /// context.
    ///
    /// # Examples
    ///
    ///     use atomicbox::AtomicOnceBox;
    ///
    ///     static GLOBAL_BOX: AtomicOnceBox<u32> = AtomicOnceBox::none();
    ///
    pub const fn none() -> Self {
        Self {
            ptr: AtomicPtr::new(null_mut()),
            phantom: PhantomData,
        }
    }

    /// Gets the value in the box, if it exists.
    pub fn get(&self, order: Ordering) -> Option<&T> {
        let ptr = self.ptr.load(order);
        unsafe { transmute(ptr) }
    }

    /// Atomically set this `AtomicOnceBox` to `value`, if it does not currently have a value.
    ///
    /// This does not allocate or free memory, and it neither clones nor drops
    /// any values.
    ///
    /// `success` must be either `Ordering::AcqRel` or `Ordering::SeqCst`,
    /// as other values would not be safe if `T` contains any data.
    ///
    /// `failure` must be either `Ordering::Acquire` or `Ordering::SeqCst`,
    /// as other values would not be safe if `T` contains any data.
    ///
    /// # Panics
    ///
    /// Panics if `order` is not one of the two allowed values.
    ///
    /// # Examples
    ///
    ///     use std::sync::atomic::Ordering;
    ///     use atomicbox::AtomicOnceBox;
    ///
    ///     let atom = AtomicOnceBox::none();
    ///     atom.try_put(Box::new("ok"), Ordering::AcqRel, Ordering::Acquire).unwrap();
    ///     let (old, new) = atom.try_put(Box::new("err"), Ordering::AcqRel, Ordering::Acquire).unwrap_err();
    ///     assert_eq!(*old, "ok");
    ///     assert_eq!(*new, "err");
    ///
    pub fn try_put(
        &self,
        value: Box<T>,
        success: Ordering,
        failure: Ordering,
    ) -> Result<&T, (&T, Box<T>)> {
        match success {
            Ordering::AcqRel | Ordering::SeqCst => {}
            _ => panic!("invalid success ordering for atomic swap"),
        }

        match failure {
            Ordering::Acquire | Ordering::SeqCst => {}
            _ => panic!("invalid failure ordering for atomic swap"),
        }

        let new_ptr = Box::into_raw(value);
        match self
            .ptr
            .compare_exchange(null_mut(), new_ptr, success, failure)
        {
            Ok(_) => Ok(unsafe { &*new_ptr }),
            Err(old_ptr) => Err(unsafe { (&*old_ptr, Box::from_raw(new_ptr)) }),
        }
    }

    /// Consume this `AtomicOnceBox`, returning the last option value it
    /// contained.
    ///
    /// # Examples
    ///
    ///     use atomicbox::AtomicOnceBox;
    ///
    ///     let atom = AtomicOnceBox::new(Some(Box::new("hello")));
    ///     assert_eq!(atom.into_inner(), Some(Box::new("hello")));
    ///
    pub fn into_inner(self) -> Option<Box<T>> {
        let last_ptr = self.ptr.load(Ordering::Acquire);
        forget(self);
        unsafe { from_ptr(last_ptr) }
    }

    /// Returns a mutable reference to the contained value.
    ///
    /// This is safe because it borrows the `AtomicOnceBox` mutably, which
    /// ensures that no other threads can concurrently access either the atomic
    /// pointer field or the boxed data it points to.
    pub fn get_mut(&mut self) -> &mut Option<Box<T>> {
        // I have a convoluted theory that Relaxed is good enough here.
        // See comment in AtomicBox::get_mut().
        let ptr = self.ptr.get_mut();
        unsafe { transmute(ptr) }
    }
}

impl<T> Drop for AtomicOnceBox<T> {
    /// Dropping an `AtomicOptionBox<T>` drops the final `Box<T>` value (if
    /// any) stored in it.
    fn drop(&mut self) {
        let last_ptr = *self.ptr.get_mut();
        unsafe {
            drop(from_ptr(last_ptr));
        }
    }
}

impl<T: Debug> Debug for AtomicOnceBox<T> {
    /// The `{:?}` format of an `AtomicOnceBox<T>` looks like
    /// `"AtomicOnceBox(MyValue)"` or `"AtomicOnceBox"`.
    fn fmt(&self, f: &mut Formatter) -> Result<(), fmt::Error> {
        if let Some(value) = self.get(Ordering::Acquire) {
            f.debug_tuple("AtomicOnceBox").field(value).finish()
        } else {
            f.write_str("AtomicOnceBox")
        }
    }
}

impl<T: Clone> Clone for AtomicOnceBox<T> {
    fn clone(&self) -> Self {
        Self::new(
            self.get(Ordering::Acquire)
                .map(|value| Box::new(value.clone())),
        )
    }
}
