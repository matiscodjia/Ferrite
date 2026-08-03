//! Où vivent les éléments d'un conteneur.
//!
//! Le cœur de calcul (`get`, `set`, les contractions, les vues) est écrit contre
//! [`Storage`], jamais contre `Box` ni contre un array concret : le lieu de
//! stockage devient un paramètre d'instanciation. Le défaut reste la pile, donc
//! une cible sans allocateur ne paie rien et le code existant ne change pas.

use crate::scalar::Scalar;
use core::ops::{Deref, DerefMut};

/// Un bloc contigu de `LEN` scalaires, de taille connue à la compilation.
///
/// Générique sur le *type de buffer* plutôt que sur un simple `NUMEL` : ça
/// couvre aussi bien `[Scalar; N]` que les arrays imbriqués (`[[Scalar; C]; R]`
/// de `Matrix`), donc un seul mécanisme de stockage pour toute la crate.
///
/// # Safety
///
/// L'implémenteur garantit que sa représentation mémoire est exactement `LEN`
/// scalaires contigus sans padding, et que le motif tout-à-zéro est une valeur
/// valide. C'est ce qui autorise [`as_flat`](Buffer::as_flat) et l'allocation
/// zéro-initialisée de `HeapStorage`.
pub unsafe trait Buffer: Sized {
    /// Nombre de scalaires du buffer, tous rangs confondus.
    const LEN: usize;

    /// Construit un buffer nul *sur place* — donc sur la pile de l'appelant.
    /// Réservé à `StackStorage` : un stockage tas ne doit jamais passer par ici.
    fn zeroed_inline() -> Self;

    fn as_flat(&self) -> &[Scalar];
    fn as_flat_mut(&mut self) -> &mut [Scalar];
}

// SAFETY: un scalaire est un scalaire contigu, et 0.0 est une valeur valide.
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

// SAFETY: un array est la juxtaposition contiguë de ses éléments, sans padding.
// Par récurrence sur l'impl de base, `[B; N]` est donc `N * B::LEN` scalaires
// contigus, et le motif tout-à-zéro reste valide.
unsafe impl<B: Buffer, const N: usize> Buffer for [B; N] {
    const LEN: usize = N * B::LEN;

    fn zeroed_inline() -> Self {
        core::array::from_fn(|_| B::zeroed_inline())
    }
    fn as_flat(&self) -> &[Scalar] {
        // SAFETY: cf. l'invariant du trait — `Self::LEN` scalaires contigus à
        // partir du début de l'array, et la durée de vie est celle de `self`.
        unsafe { core::slice::from_raw_parts(self.as_ptr() as *const Scalar, Self::LEN) }
    }
    fn as_flat_mut(&mut self) -> &mut [Scalar] {
        // SAFETY: idem, et l'emprunt exclusif de `self` interdit tout alias.
        unsafe { core::slice::from_raw_parts_mut(self.as_mut_ptr() as *mut Scalar, Self::LEN) }
    }
}

/// Un accès au buffer, quel que soit l'endroit où il vit.
///
/// N'impose rien sur la propriété : c'est ce qui laisse la place à un futur
/// stockage emprunté (buffer placé en `.bss` ou en RAM externe, sans allocateur).
pub trait Storage<B: Buffer>: Deref<Target = B> + DerefMut {}

/// Un stockage possédé, donc constructible ex nihilo.
///
/// Seul `new()` l'exige ; `get`, `set`, les vues et les contractions se
/// contentent de [`Storage`].
pub trait OwnedStorage<B: Buffer>: Storage<B> {
    fn zeroed() -> Self;
}

/// Le buffer vit inline dans la structure, donc sur la pile si la valeur est
/// possédée. Aucune dépendance à `alloc` : c'est le stockage de la cible edge,
/// et le défaut de tous les conteneurs.
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

    /// Le buffer vit sur le tas. Sonde de benchmark : permet de mesurer la
    /// montée en charge sur des tenseurs que la pile ne peut pas porter.
    /// Disparaît avec la feature `alloc`, sans que le cœur ne change d'une ligne.
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
        /// Alloue directement zéro-initialisé.
        ///
        /// Point critique : ne jamais passer par un `B` intermédiaire
        /// (`Box::new(B::zeroed_inline())` construirait le buffer sur la pile
        /// avant de le déplacer, ce qui ramènerait l'overflow qu'on corrige).
        fn zeroed() -> Self {
            let layout = Layout::new::<B>();
            if layout.size() == 0 {
                // `alloc_zeroed` interdit une taille nulle ; un buffer vide est
                // un ZST, un pointeur aligné non nul suffit.
                // SAFETY: pointeur aligné et non nul, taille nulle — la
                // représentation valide d'un ZST possédé.
                return Self(unsafe { Box::from_raw(core::ptr::NonNull::dangling().as_ptr()) });
            }
            // SAFETY: taille non nulle (testée ci-dessus) ; l'invariant de
            // `Buffer` garantit que le motif tout-à-zéro est un `B` valide, donc
            // la mémoire rendue est immédiatement initialisée.
            let ptr = unsafe { alloc_zeroed(layout) } as *mut B;
            if ptr.is_null() {
                handle_alloc_error(layout);
            }
            // SAFETY: `ptr` vient de l'allocateur global avec le layout de `B`,
            // et pointe sur un `B` valide — `Box` peut en prendre possession.
            Self(unsafe { Box::from_raw(ptr) })
        }
    }

    impl<B: Buffer> Clone for HeapStorage<B> {
        /// Copie tas → tas, sans temporaire sur la pile.
        fn clone(&self) -> Self {
            let mut copy = <Self as OwnedStorage<B>>::zeroed();
            copy.as_flat_mut().copy_from_slice(self.as_flat());
            copy
        }
    }
}
