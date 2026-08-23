//! Where a container's elements actually live.
//!
//! The compute core (`get`, `set`, contractions, views) is written against
//! [`Storage`], never against `Box` or a concrete array: the storage location
//! becomes an instantiation parameter. The default stays the stack, so a
//! target with no allocator pays nothing and existing code doesn't change.

use crate::scalar::Scalar;
use core::ops::{Deref, DerefMut};

/// A contiguous block of `LEN` scalars, of a size known at compile time.
///
/// Generic over the *buffer type* rather than a plain `NUMEL`: this covers
/// both `[Scalar; N]` and nested arrays (`[[Scalar; C]; R]`), so there is
/// a single storage mechanism for the whole crate.
///
/// # Safety
///
/// The implementer guarantees that its memory representation is exactly
/// `LEN` contiguous scalars with no padding, and that the all-zero bit
/// pattern is a valid value. This is what licenses
/// [`as_flat`](Buffer::as_flat) and `HeapStorage`'s zero-initialized
/// allocation.
pub unsafe trait Buffer: Sized {
    /// Number of scalars in the buffer, across every rank.
    const LEN: usize;

    /// Builds a zeroed buffer *in place* — so on the caller's stack.
    /// Reserved for `StackStorage`: heap storage must never go through this.
    fn zeroed_inline() -> Self;

    fn as_flat(&self) -> &[Scalar];
    fn as_flat_mut(&mut self) -> &mut [Scalar];
}

// SAFETY: a scalar is one contiguous scalar, and 0.0 is a valid value.
unsafe impl Buffer for Scalar {
    const LEN: usize = 1;

    fn zeroed_inline() -> Self {
        0.0
    }
    fn as_flat(&self) -> &[Scalar] {
        core::slice::from_ref(self)
    }
    fn as_flat_mut(&mut self) -> &mut [Scalar] {
        core::slice::from_mut(self)
    }
}

// SAFETY: an array is the contiguous juxtaposition of its elements, no
// padding. By induction on the base impl, `[B; N]` is therefore
// `N * B::LEN` contiguous scalars, and the all-zero pattern stays valid.
unsafe impl<B: Buffer, const N: usize> Buffer for [B; N] {
    const LEN: usize = N * B::LEN;

    fn zeroed_inline() -> Self {
        core::array::from_fn(|_| B::zeroed_inline())
    }
    fn as_flat(&self) -> &[Scalar] {
        // SAFETY: see the trait invariant — `Self::LEN` contiguous scalars
        // starting at the array's address, living as long as `self` does.
        unsafe { core::slice::from_raw_parts(self.as_ptr() as *const Scalar, Self::LEN) }
    }
    fn as_flat_mut(&mut self) -> &mut [Scalar] {
        // SAFETY: same as above, and the exclusive borrow of `self` rules
        // out any alias.
        unsafe { core::slice::from_raw_parts_mut(self.as_mut_ptr() as *mut Scalar, Self::LEN) }
    }
}

/// Access to the buffer, wherever it happens to live.
///
/// Imposes nothing about ownership: that's what leaves room for a future
/// borrowed storage (a buffer placed in `.bss` or external RAM, no
/// allocator).
pub trait Storage<B: Buffer>: Deref<Target = B> + DerefMut {}

/// Owned storage, so constructible out of nothing.
///
/// Only `new()` requires it; `get`, `set`, views, and contractions only
/// need [`Storage`].
pub trait OwnedStorage<B: Buffer>: Storage<B> {
    fn zeroed() -> Self;
}

/// `load_slice` error: the supplied length doesn't match `NUMEL`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LenMismatch;

/// The buffer lives inline in the struct, so on the stack if the value
/// itself is owned. No dependency on `alloc`: this is the storage for the
/// edge target, and the default for every container.
#[derive(Clone, Copy, Debug)]
pub struct StackStorage<B: Buffer>(B);

impl<B: Buffer> Deref for StackStorage<B> {
    type Target = B;
    fn deref(&self) -> &B {
        &self.0
    }
}

impl<B: Buffer> DerefMut for StackStorage<B> {
    fn deref_mut(&mut self) -> &mut B {
        &mut self.0
    }
}

impl<B: Buffer> Storage<B> for StackStorage<B> {}

impl<B: Buffer> OwnedStorage<B> for StackStorage<B> {
    fn zeroed() -> Self {
        Self(B::zeroed_inline())
    }
}

#[cfg(feature = "alloc")]
pub use heap::HeapStorage;

#[cfg(feature = "alloc")]
mod heap {
    use super::{Buffer, OwnedStorage, Storage};
    use alloc::alloc::{alloc_zeroed, handle_alloc_error, Layout};
    use alloc::boxed::Box;
    use core::ops::{Deref, DerefMut};

    /// The buffer lives on the heap. A benchmark probe: lets us measure
    /// scaling on tensors the stack cannot carry. Disappears with the
    /// `alloc` feature, without the core changing by a single line.
    #[derive(Debug)]
    pub struct HeapStorage<B: Buffer>(Box<B>);

    impl<B: Buffer> Deref for HeapStorage<B> {
        type Target = B;
        fn deref(&self) -> &B {
            &self.0
        }
    }

    impl<B: Buffer> DerefMut for HeapStorage<B> {
        fn deref_mut(&mut self) -> &mut B {
            &mut self.0
        }
    }

    impl<B: Buffer> Storage<B> for HeapStorage<B> {}

    impl<B: Buffer> OwnedStorage<B> for HeapStorage<B> {
        /// Allocates directly zero-initialized.
        ///
        /// Critical point: never go through an intermediate `B`
        /// (`Box::new(B::zeroed_inline())` would build the buffer on the
        /// stack before moving it, reintroducing the very overflow this
        /// is meant to fix).
        fn zeroed() -> Self {
            let layout = Layout::new::<B>();
            if layout.size() == 0 {
                // `alloc_zeroed` forbids a zero size; an empty buffer is a
                // ZST, a non-null aligned pointer is enough.
                // SAFETY: aligned, non-null pointer, zero size — the valid
                // representation of an owned ZST.
                return Self(unsafe { Box::from_raw(core::ptr::NonNull::dangling().as_ptr()) });
            }
            // SAFETY: non-zero size (checked above); `Buffer`'s invariant
            // guarantees the all-zero pattern is a valid `B`, so the
            // returned memory is immediately initialized.
            let ptr = unsafe { alloc_zeroed(layout) } as *mut B;
            if ptr.is_null() {
                handle_alloc_error(layout);
            }
            // SAFETY: `ptr` comes from the global allocator with `B`'s
            // layout, and points at a valid `B` — `Box` can take ownership.
            Self(unsafe { Box::from_raw(ptr) })
        }
    }

    impl<B: Buffer> Clone for HeapStorage<B> {
        /// Heap-to-heap copy, no stack temporary.
        fn clone(&self) -> Self {
            let mut copy = <Self as OwnedStorage<B>>::zeroed();
            copy.as_flat_mut().copy_from_slice(self.as_flat());
            copy
        }
    }
}
