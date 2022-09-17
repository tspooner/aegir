use concat_arrays::concat_arrays;
use std::fmt::{Debug, Display};

/// Trait for index types that can be used to access buffer elements.
pub trait Ix: Eq + Clone + Debug {
    /// Returns true if the index is a diagonal element.
    ///
    /// ```
    /// # #[macro_use] extern crate aegir;
    /// # use aegir::buffers::shapes::Ix;
    /// assert!([1, 1, 1].is_diagonal());
    /// ```
    fn is_diagonal(&self) -> bool;
}

impl Ix for () {
    fn is_diagonal(&self) -> bool { true }
}

impl Ix for usize {
    fn is_diagonal(&self) -> bool { true }
}

impl<const DIM: usize> Ix for [usize; DIM] {
    fn is_diagonal(&self) -> bool {
        let mut it = self.iter();
        let first = it.next();

        match first {
            None => true,
            Some(ix) => it.all(|jx| ix == jx),
        }
    }
}

/// Trait for types that represent the shape of a buffer.
pub trait Shape: Copy + Debug + Display {
    const DIM: usize;

    type Index: Ix;

    fn is_scalar(&self) -> bool { Self::DIM == 0 }

    fn is_vector(&self) -> bool { Self::DIM == 1 }

    fn is_matrix(&self) -> bool { Self::DIM == 2 }
}

pub type IndexOf<S> = <S as Shape>::Index;

pub trait Indices: Shape {
    type Iter: Iterator<Item = Self::Index>;

    fn indices(&self) -> Self::Iter;
}

pub trait Split: Shape + Sized {
    type Left: Concat<Self::Right>;
    type Right: Shape;

    fn split(self) -> (Self::Left, Self::Right);

    fn split_index(index: Self::Index) -> (IndexOf<Self::Left>, IndexOf<Self::Right>);
}

pub trait Concat<RHS: Shape = Self>: Shape {
    type Shape: Shape;

    fn concat(self, rhs: RHS) -> Self::Shape;

    fn concat_indices(left: Self::Index, rhs: RHS::Index) -> IndexOf<Self::Shape>;
}

pub trait DropDim: Shape + Sized {
    type Lower: Shape;

    fn drop_dim(self, dim: usize) -> Self::Lower;

    fn drop_head(self) -> Self::Lower { self.drop_dim(0) }

    fn drop_tail(self) -> Self::Lower { self.drop_dim(Self::DIM - 1) }
}

///// -----------------------------------------------------------------
///// -----------------------------------------------------------------

mod multi_product;

mod runtime;
pub use self::runtime::*;

mod compiled;
pub use self::compiled::*;
