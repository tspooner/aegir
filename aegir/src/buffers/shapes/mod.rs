//! Module for compile-time and runtime tensor shapes.
use concat_arrays::concat_arrays;
use std::fmt::{Debug, Display};

/// Error type for two incompatible buffers based on their shapes.
#[derive(Copy, Clone, Debug)]
pub struct IncompatibleShapes<S1, S2 = S1>(pub(crate) S1, pub(crate) S2)
where
    S1: Shape,
    S2: Shape;

impl<S1, S2> std::fmt::Display for IncompatibleShapes<S1, S2>
where
    S1: Shape,
    S2: Shape,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Buffer shapes are incompatible: {} vs {}.",
            self.0, self.1
        )
    }
}

impl<S1, S2> std::error::Error for IncompatibleShapes<S1, S2>
where
    S1: Shape,
    S2: Shape,
{
}

/// Trait for index types that can be used to access buffer elements.
pub trait Ix: Eq + Copy + Debug {
    /// Returns true if the index is a diagonal element.
    ///
    /// A diagonal element is defined here as an index where all components are
    /// equal.
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
    /// The dimensionality of the shape.
    const DIM: usize;

    /// Corresponding index type.
    type Index: Ix;

    type IndexIter: Iterator<Item = Self::Index>;

    fn contains(&self, ix: Self::Index) -> bool;

    /// Return an iterator over the indices of the shape.
    ///
    /// # Examples
    /// ```
    /// # use aegir::buffers::shapes::{Shape, S1};
    /// let shape: S1<5> = S1;
    /// let indices: Vec<usize> = shape.indices().collect();
    ///
    /// assert_eq!(indices, vec![0, 1, 2, 3, 4]);
    /// ```
    fn indices(&self) -> Self::IndexIter;

    /// Returns true if the shape corresponds to a scalar (DIM = 0).
    fn is_scalar(&self) -> bool { Self::DIM == 0 }

    /// Returns true if the shape corresponds to a vector (DIM = 1).
    fn is_vector(&self) -> bool { Self::DIM == 1 }

    /// Returns true if the shape corresponds to a matrix (DIM = 2).
    fn is_matrix(&self) -> bool { Self::DIM == 2 }
}

/// Type alies for index type associated with a shape.
pub type IndexOf<S> = <S as Shape>::Index;

/// Trait for splitting a shape into two symmetric parts.
///
/// __Note__: this is dualistic to [Concat] which performs the reverse
/// operation.
pub trait Split: Shape + Sized {
    /// The left side of the split.
    type Left: Concat<Self::Right, Shape = Self>;

    /// The right side of the split.
    type Right: Shape;

    /// Split the shape into two parts (left and right).
    ///
    /// # Examples
    /// ```
    /// # use aegir::buffers::shapes::{Split, S1, S2};
    /// let shape: S2<2, 5> = S2;
    ///
    /// let left: S1<2> = S1;
    /// let right: S1<5> = S1;
    ///
    /// assert_eq!(shape.split(), (left, right));
    /// ```
    fn split(self) -> (Self::Left, Self::Right);

    /// Split an index for the shape into two parts (left and right).
    fn split_index(index: Self::Index) -> (IndexOf<Self::Left>, IndexOf<Self::Right>);
}

/// Trait for concatenating two shapes into one.
///
/// __Note__: this is dualistic to [Split], which performs the reverse
/// operation.
pub trait Concat<RHS: Shape = Self>: Shape {
    type Shape: Shape;

    /// Concatenate two shapes into one.
    ///
    /// # Examples
    /// ```
    /// # use aegir::buffers::shapes::{Concat, S1, S2};
    /// let left: S1<2> = S1;
    /// let right: S1<5> = S1;
    ///
    /// let shape: S2<2, 5> = S2;
    ///
    /// assert_eq!(left.concat(right), shape);
    /// ```
    fn concat(self, rhs: RHS) -> Self::Shape;

    /// Concatenate two indices for the shape one.
    fn concat_indices(left: Self::Index, rhs: RHS::Index) -> IndexOf<Self::Shape>;
}

pub trait Zip<RHS: Shape = Self>: Shape {
    type Shape: Shape;

    fn zip(self, rhs: RHS) -> Result<Self::Shape, IncompatibleShapes<Self, RHS>>;
}

// TODO - Once impl-spec drops, we can implement this. It'd be useful for
// simplification code, much        like with operator rewrites.
// /// Trait for reducing a shape into its simplest form.
// ///
// /// This typically involves trimming either end of unitary values.
// pub trait Reduce: Shape {
// type Reduced: Shape;

// /// Trim the shape.
// fn reduce(self) -> Self::Reduced;
// }

mod multi_product;

mod runtime;
pub use self::runtime::*;

mod compiled;
pub use self::compiled::*;
