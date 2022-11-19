use super::*;
use std::ops::{Index, IndexMut};

/// Fixed rank, variable `DIM`-dimensional shape.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct SDynamic<const DIM: usize>(pub [usize; DIM]);

impl<const DIM: usize> Index<usize> for SDynamic<DIM> {
    type Output = usize;

    fn index(&self, idx: usize) -> &usize { self.0.index(idx) }
}

impl<const DIM: usize> IndexMut<usize> for SDynamic<DIM> {
    fn index_mut(&mut self, idx: usize) -> &mut usize { self.0.index_mut(idx) }
}

impl<const DIM: usize> Shape for SDynamic<DIM> {
    type Index = [usize; DIM];
    type IndexIter = multi_product::MultiProduct<DIM>;

    const DIM: usize = DIM;

    fn contains(&self, ix: [usize; DIM]) -> bool {
        ix.iter().zip(self.0.iter()).all(|(l, r)| l < r)
    }

    fn indices(&self) -> Self::IndexIter { multi_product::MultiProduct::new(self.0) }
}

impl<const DIM: usize> std::fmt::Display for SDynamic<DIM> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SDynamic({:?})", self.0)
    }
}

macro_rules! impl_add_dim {
    ($n:literal + $m:literal) => {
        impl Concat<SDynamic<$m>> for SDynamic<$n> {
            type Shape = SDynamic<{$n + $m}>;

            fn concat(self: SDynamic<$n>, rhs: SDynamic<$m>) -> Self::Shape {
                SDynamic(concat_arrays!(self.0, rhs.0))
            }

            fn concat_indices(left: [usize; $n], rhs: [usize; $m]) -> IndexOf<Self::Shape> {
                concat_arrays!(left, rhs)
            }
        }
    };
    ([$($n:literal),*] + $m:literal) => {
        $(impl_add_dim!($n + $m);)*
    };
    ($ns:tt + [$($m:literal),*]) => {
        $(impl_add_dim!($ns + $m);)*
    }
}

impl_add_dim!([0, 1, 2, 3, 4, 5, 6] + [0, 1, 2, 3, 4, 5, 6]);

// macro_rules! impl_drop_dim {
// ($n:literal) => {
// impl DropDim for SDynamic<$n> {
// type Lower = SDynamic<{$n - 1}>;

// fn drop_dim(self, dim: usize) -> SDynamic<{$n - 1}> {
// if dim >= Self::DIM { panic!("Lame..."); }

// SDynamic(array_init::array_init(|i| if i >= dim { self.0[i + 1] } else {
// self.0[i] })) }
// }
// };
// ($($n:literal),*) => {
// $(impl_drop_dim!($n);)*
// };
// }

// impl_drop_dim!(1, 2, 3, 4, 5, 6);
