//! Module for compile-time and runtime tensor shapes.
use concat_arrays::concat_arrays;
use std::fmt::{Debug, Display};

/// Error type for two incompatible buffers based on their shapes.
#[derive(Copy, Clone, Debug)]
pub struct IncompatibleShapes<L: Shape, R: Shape = L> {
    pub left: L,
    pub right: R,
}

impl<L: Shape, R: Shape> IncompatibleShapes<L, R> {
    pub fn reverse(self) -> IncompatibleShapes<R, L> {
        IncompatibleShapes {
            left: self.right,
            right: self.left,
        }
    }
}

impl<L: Shape, R: Shape> std::fmt::Display for IncompatibleShapes<L, R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Buffer shapes are incompatible: {} vs {}.",
            self.left, self.right
        )
    }
}

impl<L: Shape, R: Shape> std::error::Error for IncompatibleShapes<L, R> {}

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

pub trait Shaped {
    /// [Shape](shapes::Shape) associated with this type.
    type Shape: Shape;

    /// Return the [Shape](Buffer::Shape) associated with the type.
    ///
    /// # Examples
    /// ```
    /// # use aegir::buffers::{Buffer, shapes::S2};
    /// // Take the 2x2 identity matrix...
    /// let buffer = [
    ///     [1.0, 0.0],
    ///     [0.0, 1.0]
    /// ];
    ///
    /// // We can assert that the row/col counts match by the following type annotation:
    /// let shape: S2<2, 2> = buffer.shape();
    /// ```
    fn shape(&self) -> Self::Shape;
}

/// Type shortcut for the [Shape] associated with a [Shaped].
pub type ShapeOf<B> = <B as Shaped>::Shape;

/// Trait for types that represent the shape of a buffer.
pub trait Shape: Copy + Debug + Display {
    /// The dimensionality of the shape.
    const DIM: usize;

    /// Corresponding index type.
    type Index: Ix;

    type IndexIter: Iterator<Item = Self::Index>;

    fn contains(&self, ix: Self::Index) -> bool;

    fn cardinality(&self) -> usize;

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

mod multi_product;

mod runtime;
pub use self::runtime::*;

mod compiled;
pub use self::compiled::*;
