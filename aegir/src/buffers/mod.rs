//! Module for tensor abstractions and their implementations.
//!
//! `aegir` is designed to be generic over the concrete [Buffer] types used by
//! the operators. This has the advantage of flexibility, but also exposes
//! performance advantages; fixed-length arrays, for example, can be optmised
//! much more aggressively in the LLVM compared with arbitrary-length
//! vectors. This does, however, add some complexity.
//!
//! # Type Hierarchy
//! The type hierarchy exposed by [aegir::buffers] can broadly be split into
//! three parts: [classes](Class), [buffers](Buffer) and [scalars](Scalar). Each
//! [buffer](Buffer) is a homogeneous collection of numerical values with a
//! given [shape](Buffer::Shape) and underlying [field](Buffer::Field). A
//! [scalar](Scalar) type is a special case of a [buffer](Buffer) in
//! which the [field](Buffer::Field) is equal to the implementing type itself.
//! For example, [Scalar] is automatically derived for the [f64] primitive since
//! it supports base numerical operations (addition, subtraction, etc...), but
//! also implements [Buffer] with [Buffer::Field] assigned to [f64]. In other
//! words, the [Scalar] trait defines a fixed point of the type hierarchy.
//! Indeed, since the associated type [Buffer::Field] has a [Scalar] constraint,
//! we get a nice uniqueness property and a unilateral support of [Buffer] at
//! all levels. Now we understand the relationship between [Buffer] and
//! [Scalar], it remains only to explain the purpose of [Class].
//!
//! #### Buffer Classes
//! Types that implement the [Class] trait are generally used to form a semantic
//! grouping over [buffers](Buffer). For example, [Arrays] groups together all
//! fixed-length arrays under the same natural umbrella. A given implementation
//! of [Class] then asserts a _unique mapping_ between a shape-field pair, and a
//! concrete [Buffer] type (with compatible associated types) within the context
//! of a particular grouping. In other words, there can be only one type
//! associated with [Arrays] that has said shape and field, and this type can be
//! reached via `Class<S, F>::Buffer`.
//!
//! There are two key reasons for this construction:
//! 1. [Classes](Class) afford us the ability to construct new buffers with a
//! particular shape and    field without knowing the concrete [Buffer] type.
//! The yields much greater flexibility    and allows us to implement many of
//! the core "source" types with much greater generality.    For more
//! information, see e.g. [Class::build]. 2. Because `Class<S, F>::Buffer` is
//! unique, we can define precedence relations over    [classes](Class). For
//! example, dynamically allocated arrays should take precedence over
//!    fixed-length arrays to ensure the widest level of runtime compatibility.
//! This allows us to    mix-and-match our data structures and, say, perform an
//! inner product between `[f64; 2]` and    `(f64, f64)`.
pub mod shapes;

use shapes::{Concat, Ix, Shape};
use num_traits::{One, Zero};

/// Marker trait for class subscriptions of [Buffer] types.
///
/// This trait is used to define a semantic grouping over buffer types.
/// For a given class, implementing this trait defines a unique
/// mapping between a [Shape](Buffer::Shape) and [Field](Buffer::Field),
/// and a (sized) [Buffer] type.
///
/// __Note:__ as [Class::Buffer] must be sized, one can _always_ leverage the
/// [OwnedOf] type alias to recover a concrete, owned type from a given buffer.
/// This is particularly useful to establish the return type of methods such as
/// [Buffer::map].
///
/// # Examples
/// The Class trait is particularly useful for constructing instances of
/// buffer types. In the example below, we consider the [Arrays] class
/// which subsumes all compile-time homogeneous data buffers. Note that we
/// never explicitly pass numeric values to [S1](shapes::S1) or
/// [S2](shapes::S2), and instead leave it to the compiler to infer.
/// ```
/// # #[macro_use] extern crate aegir;
/// # use aegir::buffers::{Class, Arrays, shapes::{S1, S2}};
/// assert_eq!(Arrays::full(S1, 1.0f64), [1.0, 1.0, 1.0]);
/// assert_eq!(Arrays::full(S2, 0.0f64), [[0.0], [0.0]]);
/// ```
///
/// It thus follows that the snippet below is invalid, as the target is
/// only 1-dimensional:
/// ```compile_fail
/// # #[macro_use] extern crate aegir;
/// # use aegir::buffers::{Class, Arrays, shapes::{S1, S2}};
/// assert_eq!(Arrays::full(S2, 1.0f64), [1.0, 1.0, 1.0]);
/// ```
pub trait Class<S: Shape> {
    /// The associated buffer types.
    type Buffer<F: Scalar>: Buffer<Class = Self, Field = F> + Sized;

    /// Construct a [Buffer](Class::Buffer) using a function over indices.
    ///
    /// This method is good for general purpose isntantiation of buffers, but it
    /// is not guaranteed to be efficient as it performs a full element-wise
    /// scan of the buffer.
    ///
    /// # Examples
    /// ```
    /// # #[macro_use] extern crate aegir;
    /// # use aegir::buffers::{Class, Arrays, shapes::{S1, S2}};
    /// const R: usize = 3;
    /// const C: usize = 3;
    ///
    /// fn ind2val(index: [usize; 2]) -> usize {
    ///     index[0] * R + index[1]
    /// }
    ///
    /// // The type annotation isn't actually necessary, but it does improve readability.
    /// let buffer: [[usize; C]; R] = Arrays::build(S2, ind2val);
    ///
    /// assert_eq!(buffer, [
    ///     [0, 1, 2],
    ///     [3, 4, 5],
    ///     [6, 7, 8]
    /// ]);
    /// ```
    fn build<F: Scalar>(shape: S, f: impl Fn(S::Index) -> F) -> Self::Buffer<F>;

    /// Construct a [Buffer](Class::Buffer) with default value and subset
    /// assignment.
    ///
    /// This method is generally more efficient than [Class::build] as the scan
    /// is only performed over the subset of entries provided.
    ///
    /// # Examples
    /// ```
    /// # #[macro_use] extern crate aegir;
    /// # use aegir::buffers::{Class, Arrays, shapes::{S2, Shape}};
    /// const R: usize = 3;
    /// const C: usize = 3;
    ///
    /// let shape: S2<R, C> = S2;
    /// let indices = shape.indices().filter(|index| {
    ///     let row_extremal = index[0] == 0 || index[0] == R - 1;
    ///     let col_extremal = index[1] == 0 || index[1] == C - 1;
    ///
    ///     row_extremal && col_extremal
    /// });
    ///
    /// // The type annotation isn't actually necessary, but it does improve readability.
    /// let buffer: [[usize; C]; R] = Arrays::build_subset(S2, 0, indices, |[r, c]| {
    ///     r + c
    /// });
    ///
    /// assert_eq!(buffer, [
    ///     [0, 0, 2],
    ///     [0, 0, 0],
    ///     [2, 0, 4]
    /// ]);
    /// ```
    fn build_subset<F: Scalar>(
        shape: S,
        base: F,
        subset: impl Iterator<Item = shapes::IndexOf<S>>,
        active: impl Fn(S::Index) -> F,
    ) -> Self::Buffer<F>;

    /// Construct a [Buffer](Class::Buffer) and populate with a given value.
    ///
    /// This method can be very efficiently implemented for many [Buffer] types.
    /// For example, Rust comes equipped with fast initialisation of arrays
    /// which avoids the need for an explicit elementwise scan. This method
    /// should be preferred if you are sure each element should be
    /// initialised with the same value.
    ///
    /// # Examples
    /// This method is most commonly used to initialise construct a buffer of
    /// all zeroes or all ones.
    /// ```
    /// # #[macro_use] extern crate aegir;
    /// # use aegir::buffers::{Class, Tuples, shapes::{S1, Shape}};
    /// assert_eq!(Tuples::full(S1, 0.0), (0.0, 0.0));
    /// assert_eq!(Tuples::full(S1, 1.0), (1.0, 1.0));
    /// ```
    fn full<F: Scalar>(shape: S, value: F) -> Self::Buffer<F> { Self::build(shape, |_| value) }

    fn zeroes<F: Scalar>(shape: S) -> Self::Buffer<F> { Self::full(shape, F::zero()) }

    fn ones<F: Scalar>(shape: S) -> Self::Buffer<F> { Self::full(shape, F::one()) }

    /// Construct a zeroed [Buffer](Class::Buffer) with a given value along the
    /// diagonal.
    ///
    /// This method works by initialising only the diagonal entries with a given
    /// value. Typically, this is used akin to the "eye" method found in
    /// most scientific computing/linear-algebra libraries. It is worth
    /// noting that the [Ix](shapes::Ix) trait includes a method
    /// [Ix::is_diagonal](shapes::Ix::is_diagonal) which can be used to
    /// check if a given index lies on the diagonal.
    ///
    /// __Note:__ this method uses [Class::build] by default, so be sure to
    /// override.
    ///
    /// # Examples
    /// ```
    /// # #[macro_use] extern crate aegir;
    /// # use aegir::buffers::{Class, Arrays, shapes::{S2, Shape}};
    /// assert_eq!(Arrays::diagonal(S2, 5.0), [
    ///     [5.0, 0.0],
    ///     [0.0, 5.0]
    /// ]);
    /// ```
    fn diagonal<F: Scalar>(shape: S, value: F) -> Self::Buffer<F> {
        Self::build(shape, |ijk: S::Index| {
            if ijk.is_diagonal() {
                value
            } else {
                num_traits::zero()
            }
        })
    }

    /// Construct a zeroed [Buffer](Class::Buffer) with ones along the diagonal.
    ///
    /// An identity buffer is simply intitialised with all zeroes, and the value
    /// one assigned to each diagonal entry according to
    /// [Ix::is_diagonal](shapes::Ix::is_diagonal).
    ///
    /// __Note:__ this method uses [Class::diagonal] by default, but there may
    /// be more efficient implementations for a given type. Be sure to
    /// override if this is the case.
    ///
    /// # Examples
    /// ```
    /// # #[macro_use] extern crate aegir;
    /// # use aegir::buffers::{Class, Arrays, shapes::{S2, Shape}};
    /// let eye: [[f64; 2]; 2] = Arrays::identity(S2);
    ///
    /// assert_eq!(eye, [
    ///     [1.0, 0.0],
    ///     [0.0, 1.0]
    /// ]);
    /// ```
    fn identity<F: Scalar>(shape: S) -> Self::Buffer<F> { Self::diagonal(shape, num_traits::one()) }
}

/// Type shortcut for the [Class] associated with a [Buffer].
pub type BufferOf<C, S, F> = <C as Class<S>>::Buffer<F>;

/// Trait for types defining a data buffer over a fixed [field](Buffer::Field)
/// and [shape](Buffer::Shape).
pub trait Buffer {
    /// [Class] associated with the buffer.
    type Class: Class<Self::Shape>;

    /// [Shape](shapes::Shape) associated with the buffer.
    type Shape: Shape;

    /// [Scalar] field associated with the buffer.
    type Field: Scalar;

    /// Return the [Shape](Buffer::Shape) of the buffer.
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

    /// Return the value of the buffer at index `ix`.
    ///
    /// # Examples
    /// ```
    /// # use aegir::buffers::{Buffer, shapes::S2};
    /// let buffer = [
    ///     [1.0, 0.0],
    ///     [0.0, 1.0]
    /// ];
    ///
    /// assert_eq!(buffer.get([0, 0]), Some(1.0));
    /// assert_eq!(buffer.get([0, 1]), Some(0.0));
    /// assert_eq!(buffer.get([1, 0]), Some(0.0));
    /// assert_eq!(buffer.get([1, 1]), Some(1.0));
    /// ```
    fn get(&self, ix: shapes::IndexOf<Self::Shape>) -> Option<Self::Field> {
        if self.shape().contains(ix) {
            Some(self.get_unchecked(ix))
        } else {
            None
        }
    }

    /// Return the value of the buffer at index `ix`.
    ///
    /// # Examples
    /// ```
    /// # use aegir::buffers::{Buffer, shapes::S2};
    /// let buffer = [
    ///     [1.0, 0.0],
    ///     [0.0, 1.0]
    /// ];
    ///
    /// assert_eq!(buffer.get([0, 0]), Some(1.0));
    /// assert_eq!(buffer.get([0, 1]), Some(0.0));
    /// assert_eq!(buffer.get([1, 0]), Some(0.0));
    /// assert_eq!(buffer.get([1, 1]), Some(1.0));
    /// ```
    fn get_unchecked(&self, ix: shapes::IndexOf<Self::Shape>) -> Self::Field;

    /// Perform an element-wise transformation of the buffer (in-place).
    ///
    /// # Examples
    /// ```
    /// # use aegir::buffers::Buffer;
    /// let buffer = vec![0.0, 1.0, 2.0, 3.0];
    /// let new_buffer = buffer.map(|el| el * 2.0);
    ///
    /// assert_eq!(new_buffer[0], 0.0);
    /// assert_eq!(new_buffer[1], 2.0);
    /// assert_eq!(new_buffer[2], 4.0);
    /// assert_eq!(new_buffer[3], 6.0);
    /// ```
    fn map<F: Scalar, M: Fn(Self::Field) -> F>(self, f: M) -> OwnedOf<Self, F>
    where
        Self: Sized,
    {
        <Self::Class as Class<Self::Shape>>::build(self.shape(), |ix| f(self.get_unchecked(ix)))
    }

    /// Perform an element-wise transformation of the buffer (reference).
    fn map_ref<F: Scalar, M: Fn(Self::Field) -> F>(&self, f: M) -> OwnedOf<Self, F> {
        <Self::Class as Class<Self::Shape>>::build(self.shape(), |ix| f(self.get_unchecked(ix)))
    }

    /// Perform a fold over the elements of the buffer.
    ///
    /// # Examples
    /// ```
    /// # use aegir::buffers::Buffer;
    /// let buffer = vec![0.0, 1.0, 2.0, 3.0];
    ///
    /// assert_eq!(buffer.fold(0.0, |init, el| init + 2.0 * el), 12.0);
    /// ```
    fn fold<F, M: Fn(F, Self::Field) -> F>(&self, init: F, f: M) -> F {
        self.shape().indices().fold(init, |acc, ix| f(acc, self.get_unchecked(ix)))
    }

    /// Sum over the elements of the buffer.
    ///
    /// # Examples
    /// ```
    /// # use aegir::buffers::Buffer;
    /// let buffer = vec![0.0, 1.0, 2.0, 3.0];
    ///
    /// assert_eq!(buffer.sum(), 6.0);
    /// assert_eq!(buffer.map(|el| 2.0 * el).sum(), 12.0);
    /// ```
    fn sum(&self) -> Self::Field { self.fold(num_traits::zero(), |init, el| init + el) }

    /// Create an owned instance from a borrowed buffer, usually by cloning.
    fn to_owned(&self) -> OwnedOf<Self>;

    /// Convert buffer directly into an owned instance, cloning when necessary.
    fn into_owned(self) -> OwnedOf<Self>;

    /// Create a [Constant](crate::sources::Constant) source node from the
    /// buffer, usually by cloning.
    fn to_constant(&self) -> crate::sources::Constant<OwnedOf<Self>> {
        crate::sources::Constant(self.to_owned())
    }

    /// Convert buffer directly into a [Constant](crate::sources::Constant)
    /// source node.
    fn into_constant(self) -> crate::sources::Constant<OwnedOf<Self>>
    where
        Self: Sized,
    {
        crate::sources::Constant(self.into_owned())
    }
}

/// Type shortcut for the owned variant of a [Buffer].
pub type OwnedOf<B, F = <B as Buffer>::Field> = <
    <B as Buffer>::Class as Class<<B as Buffer>::Shape>
>::Buffer<F>;

/// Type shortcut for the [Shape] associated with a [Buffer].
pub type ShapeOf<B> = <B as Buffer>::Shape;

/// Type shortcut for the [Field] associated with a [Buffer].
pub type FieldOf<B> = <B as Buffer>::Field;

/// Type shortcut for the [Class] associated with a [Buffer].
pub type ClassOf<B> = <B as Buffer>::Class;

/// Error type for two incompatible buffers based on their shapes.
#[derive(Copy, Clone, Debug)]
pub struct IncompatibleShapes<S1, S2 = S1>(pub(crate) S1, pub(crate) S2)
where
    S1: shapes::Shape,
    S2: shapes::Shape;

impl<S1, S2> std::fmt::Display for IncompatibleShapes<S1, S2>
where
    S1: shapes::Shape,
    S2: shapes::Shape,
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
    S1: shapes::Shape,
    S2: shapes::Shape,
{
}

/// Trait for performing a zip and fold over a pair of buffers.
pub trait ZipFold<RHS: Buffer = Self>: Buffer {
    /// Perform a zip and fold over a pair of buffers.
    ///
    /// # Examples
    /// ```
    /// # use aegir::buffers::ZipFold;
    /// let b1 = [1.0, 2.0, 3.0];
    /// let b2 = [-1.0, 0.0, 1.0];
    ///
    /// assert_eq!(b1.zip_fold(&b2, 0.0, |acc, (l, r)| acc + l * r).unwrap(), 2.0);
    /// ```
    fn zip_fold<F: Scalar, M: Fn(F, (Self::Field, RHS::Field)) -> F>(
        &self,
        rhs: &RHS,
        init: F,
        f: M,
    ) -> Result<F, IncompatibleShapes<Self::Shape, RHS::Shape>>;
}

/// Trait for combining two buffers in an elementwise fashion.
pub trait ZipMap<RHS: Buffer = Self>: Buffer {
    type Output<F: Scalar>: Buffer<Field = F>;

    /// Combine two buffers in an elementwise fashion.
    ///
    /// # Examples
    /// ```
    /// # use aegir::buffers::ZipMap;
    /// let b1 = [1.0, 2.0, 3.0];
    /// let b2 = [-1.0, 0.0, 1.0];
    ///
    /// assert_eq!(b1.zip_map(&b2, |l, r| l * r).unwrap(), [-1.0, 0.0, 3.0]);
    /// ```
    fn zip_map<F: Scalar, M: Fn(Self::Field, RHS::Field) -> F>(
        self,
        rhs: &RHS,
        f: M
    ) -> Result<Self::Output<F>, IncompatibleShapes<Self::Shape, RHS::Shape>>
    where
        Self: Sized,
    {
        self.zip_map_ref(rhs, f)
    }

    /// Combine two buffers in an elementwise fashion.
    ///
    /// # Examples
    /// ```
    /// # use aegir::buffers::ZipMap;
    /// let b1 = [1.0, 2.0, 3.0];
    /// let b2 = [-1.0, 0.0, 1.0];
    ///
    /// assert_eq!(b1.zip_map(&b2, |l, r| l * r).unwrap(), [-1.0, 0.0, 3.0]);
    /// ```
    fn zip_map_ref<F: Scalar, M: Fn(Self::Field, RHS::Field) -> F>(
        &self,
        rhs: &RHS,
        f: M,
    ) -> Result<Self::Output<F>, IncompatibleShapes<Self::Shape, RHS::Shape>>;
}

pub trait Contract<RHS: Buffer<Field = Self::Field>, const AXES: usize = 1>: Buffer {
    type Output: Buffer<Field = Self::Field>;

    fn contract(self, rhs: RHS) -> Result<Self::Output, IncompatibleShapes<Self::Shape, RHS::Shape>>;
}

mod scalars;
pub use scalars::*;

mod tuples;
pub use tuples::*;

mod vecs;
pub use vecs::*;

mod arrays;
pub use arrays::*;

// #[cfg(feature = "ndarray")]
// mod ndarray;
// #[cfg(feature = "ndarray")]
// pub use self::ndarray::*;
