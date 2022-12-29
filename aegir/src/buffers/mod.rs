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
//! buffer is a homogeneous collection of numerical values with a
//! given shape and underlying scalar field. A
//! scalar type is a special case of a buffer in
//! which the field is equal to the implementing type itself.
//! For example, [Scalar] is automatically derived for the [f64] primitive since
//! it supports base numerical operations (addition, subtraction, etc...), but
//! also implements [Buffer] with [Buffer::Field] assigned to [f64]. In other
//! words, the [Scalar] trait defines a fixed point of the type hierarchy.
//! Indeed, since the associated type [Buffer::Field] has a [Scalar] constraint,
//! we get a uniqueness property and a unilateral support of [Buffer] at
//! all levels. Now we understand the relationship between [Buffer] and
//! [Scalar], it remains only to explain the purpose of [Class].
//!
//! #### Buffer Classes
//! Types that implement the [Class] trait are generally used to form a semantic
//! grouping over buffers. For example, [Arrays] groups together all
//! fixed-length arrays under the same natural umbrella. A given implementation
//! of [Class] then asserts a _unique mapping_ between a shape-field pair, and a
//! concrete [Buffer] type (with compatible associated types) within the context
//! of a particular grouping. In other words, there can be only one type
//! associated with [Arrays] that has said shape and field, and this type can be
//! reached via `Class<S, F>::Buffer`.
//!
//! #### Advantages
//! Classes afford us the ability to construct new buffers with a
//! particular shape and field without knowing the concrete [Buffer] type.
//! This yields much greater flexibility and allows us to implement many of
//! the core "source" types with much greater generality. For more
//! information, see e.g. [Class::build].
pub mod shapes;

use shapes::{Ix, Shape, IncompatibleShapes, Broadcast, BShape};

// ---------------------------------------------------------------------------
// Core Trait Definitions
// ---------------------------------------------------------------------------
/// Marker trait for class subscriptions of [Buffer] types.
///
/// This trait is used to define a semantic grouping over buffer types. For a given class,
/// implementing this trait defines a unique mapping between a shape and field, and a (sized)
/// buffer type.
///
/// # Examples
///
/// The Class trait is particularly useful for constructing instances of
/// buffer types. In the example below, we consider the [Arrays] class
/// which subsumes all compile-time homogeneous data buffers. Note that we
/// never explicitly pass numeric values to [S1](shapes::S1) or
/// [S2](shapes::S2), and instead leave it to the compiler to infer.
///
/// ```
/// # #[macro_use] extern crate aegir;
/// # use aegir::buffers::{Class, Arrays, shapes::{S1, S2}};
/// assert_eq!(Arrays::full(S1, 1.0f64), [1.0, 1.0, 1.0]);
/// assert_eq!(Arrays::full(S2, 0.0f64), [[0.0], [0.0]]);
/// ```
///
/// It thus follows that the snippet below is invalid, as the target is
/// only 1-dimensional:
///
/// ```compile_fail
/// # #[macro_use] extern crate aegir;
/// # use aegir::buffers::{Class, Arrays, shapes::{S1, S2}};
/// assert_eq!(Arrays::full(S2, 1.0f64), [1.0, 1.0, 1.0]);
/// ```
pub trait Class<S: Shape> {
    /// The associated buffer types.
    type Buffer<F: Scalar>: Buffer<Class = Self, Shape = S, Field = F>;

    /// Construct a [Buffer](Class::Buffer) using a function over indices.
    ///
    /// This method is good for general purpose isntantiation of buffers, but it
    /// is not guaranteed to be efficient as it performs a full element-wise
    /// scan of the buffer.
    ///
    /// # Examples
    ///
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
    ///
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
    ///
    /// This method is most commonly used to initialise construct a buffer of
    /// all zeroes or all ones.
    ///
    /// ```
    /// # #[macro_use] extern crate aegir;
    /// # use aegir::buffers::{Class, Tuples, shapes::{S1, Shape}};
    /// assert_eq!(Tuples::full(S1, 0.0), (0.0, 0.0));
    /// assert_eq!(Tuples::full(S1, 1.0), (1.0, 1.0));
    /// ```
    fn full<F: Scalar>(shape: S, value: F) -> Self::Buffer<F> { Self::build(shape, |_| value) }

    /// Construct a [Buffer](Class::Buffer) and populate with zeroes.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate aegir;
    /// # use aegir::buffers::{Class, Tuples, shapes::{S1, Shape}};
    /// assert_eq!(Tuples::zeroes(S1), Tuples::full(S1, 0.0));
    /// ```
    fn zeroes<F: Scalar>(shape: S) -> Self::Buffer<F> { Self::full(shape, F::zero()) }

    /// Construct a [Buffer](Class::Buffer) and populate with ones.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate aegir;
    /// # use aegir::buffers::{Class, Tuples, shapes::{S1, Shape}};
    /// assert_eq!(Tuples::ones(S1), Tuples::full(S1, 1.0));
    /// ```
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
    ///
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
    fn identity<F: Scalar>(shape: S) -> Self::Buffer<F> {
        Self::diagonal(shape, num_traits::one())
    }
}

/// Type alias for [Class::Buffer].
pub type BufferOf<C, S, F> = <C as Class<S>>::Buffer<F>;

/// Trait for container types that have a fixed shape and scalar field.
pub trait Buffer: Clone + shapes::Shaped + IntoSpec<Buffer = Self> {
    /// [Class] associated with the buffer.
    type Class: Class<Self::Shape, Buffer<Self::Field> = Self>;

    /// [Scalar] field associated with the buffer.
    type Field: Scalar;

    /// Return the class associated with the buffer.
    fn class() -> Self::Class;

    /// Return the value of the buffer at index `ix`.
    ///
    /// # Examples
    ///
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
    ///
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

    /// Perform an element-wise transformation of the buffer.
    ///
    /// # Examples
    ///
    /// ```
    /// # use aegir::buffers::Buffer;
    /// let buffer = vec![0.0, 1.0, 2.0, 3.0];
    /// let new_buffer = buffer.map(|x| x as u8 * 2);
    ///
    /// assert_eq!(new_buffer[0], 0);
    /// assert_eq!(new_buffer[1], 2);
    /// assert_eq!(new_buffer[2], 4);
    /// assert_eq!(new_buffer[3], 6);
    /// ```
    fn map<F: Scalar, M: Fn(Self::Field) -> F>(
        self,
        f: M,
    ) -> <Self::Class as Class<Self::Shape>>::Buffer<F>;

    /// Perform an element-wise transformation of the buffer reference.
    fn map_ref<F: Scalar, M: Fn(Self::Field) -> F>(
        &self,
        f: M,
    ) -> <Self::Class as Class<Self::Shape>>::Buffer<F> {
        <Self::Class as Class<Self::Shape>>::build(self.shape(), |ix| f(self.get_unchecked(ix)))
    }

    /// Perform an element-wise transformation of the buffer in-place.
    ///
    /// # Examples
    ///
    /// ```
    /// # use aegir::buffers::Buffer;
    /// let mut buffer = vec![0.0, 1.0, 2.0, 3.0];
    ///
    /// buffer.mutate(|el| -el);
    ///
    /// assert_eq!(buffer[0], -0.0);
    /// assert_eq!(buffer[1], -1.0);
    /// assert_eq!(buffer[2], -2.0);
    /// assert_eq!(buffer[3], -3.0);
    /// ```
    fn mutate<M: Fn(Self::Field) -> Self::Field>(&mut self, f: M);

    /// Perform a fold over the elements of the buffer.
    ///
    /// # Examples
    ///
    /// ```
    /// # use aegir::buffers::Buffer;
    /// let buffer = vec![0.0, 1.0, 2.0, 3.0];
    ///
    /// assert_eq!(buffer.fold(0.0, |init, el| init + 2.0 * el), 12.0);
    /// ```
    fn fold<F, M: Fn(F, Self::Field) -> F>(&self, init: F, f: M) -> F {
        self.shape()
            .indices()
            .fold(init, |acc, ix| f(acc, self.get_unchecked(ix)))
    }

    /// Sum over the elements of the buffer.
    ///
    /// # Examples
    ///
    /// ```
    /// # use aegir::buffers::Buffer;
    /// let buffer = vec![0.0, 1.0, 2.0, 3.0];
    ///
    /// assert_eq!(buffer.sum(), 6.0);
    /// assert_eq!(buffer.map(|el| 2.0 * el).sum(), 12.0);
    /// ```
    fn sum(&self) -> Self::Field { self.fold(num_traits::zero(), |init, el| init + el) }

    /// Convert buffer directly into a [Constant](crate::meta::Constant)
    /// source node.
    fn into_constant(self) -> crate::meta::Constant<Self> { crate::meta::Constant(self) }
}

/// Type alias for [Buffer::Field].
pub type FieldOf<B> = <B as Buffer>::Field;

/// Type alias for [Buffer::Class].
pub type ClassOf<B> = <B as Buffer>::Class;

// ---------------------------------------------------------------------------
// Zip-based Operations
// ---------------------------------------------------------------------------
// TODO - Implement symmetry in these traits.
//
// Implementing symmetry in this operation can be done via the following:
//  1. Use the shapes::Zip trait as the unified source of truth for the output
// shape.  2. Replace Self::Output with Self::ZipClass; i.e. the user
// defines only the __Class__ of     the resulting buffer, not the field or
// shape. Those will instead be implied by Zip and     by the closure passed
// to the function.  3. Add a trait bound such that RHS also implements
// ZipMap, and has identical ZipClass.
//
// This works as of v1.65 - yay! - but unfortunately we end up polluting where
// clauses extensively downstream - nay! For now we will just leave it as
// unconstrained and leave it to the user to handle that. However, once
// RFC-2089 (implied-bounds) lands, we should be able to revisit this and
// potentially enforce symmetry.
/// Trait for performing a zip and fold over a pair of buffers.
pub trait ZipFold<RHS: Buffer = Self>: Buffer {
    /// Perform a zip and fold over a pair of buffers.
    ///
    /// # Examples
    ///
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

/// Trait for combining a pair of buffers in an elementwise fashion.
pub trait ZipMap<RHS: Buffer = Self>: Buffer
where
    Self::Shape: Broadcast<RHS::Shape>,
{
    /// The generic container type associated with the output.
    type Output<F: Scalar>: Buffer<Field = F, Shape = BShape<Self::Shape, RHS::Shape>>;

    /// Combine two buffers elementwise, yielding a new buffer.
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
        f: M,
    ) -> Result<Self::Output<F>, IncompatibleShapes<Self::Shape, RHS::Shape>>;

    /// Combine two buffers elementwise, yielding a buffer with the same `Buffer::Field`.
    ///
    /// This method should be preferred over `ZipMap::zip_map` when the transformation
    /// does not change the underlying field type. This is because the type signature
    /// allows the implementor to perform in-place mutations rather instantiate new
    /// buffers in memory.
    #[inline]
    fn zip_map_id<M: Fn(Self::Field, RHS::Field) -> Self::Field>(
        self,
        rhs: &RHS,
        f: M,
    ) -> Result<Self::Output<Self::Field>, IncompatibleShapes<Self::Shape, RHS::Shape>> {
        self.zip_map(rhs, f)
    }

    /// Expand a buffer into the corresponding shape given `RHS`.
    ///
    /// This method is useful in `Spec` optimisations in which the transform
    /// is simply a direct pass through. As in the example below, we often
    /// have cases like `x * 1 = x` for which we simply want to perform a type
    /// translation from input to output types, or a broadcast.
    ///
    /// # Examples
    ///
    /// ```
    /// # use aegir::buffers::{ZipMap, shapes::S1};
    /// let buf = 1.0;
    /// let out = <f64 as ZipMap<[f64; 3]>>::zip_shape(buf, S1).unwrap();
    ///
    /// assert_eq!(out, [1.0, 1.0, 1.0]);
    /// ```
    fn zip_shape(self, rshape: RHS::Shape) -> Result<Self::Output<Self::Field>, IncompatibleShapes<Self::Shape, RHS::Shape>>;
}

/// Trait for combining two buffers elementwise and in-place.
pub trait ZipMut<RHS: Buffer = Self>: Buffer {
    /// Combine two buffers elementwise by mutating `self` in-place.
    fn zip_mut<M: Fn(Self::Field, RHS::Field) -> Self::Field>(
        &mut self,
        rhs: &RHS,
        f: M,
    ) -> Result<(), IncompatibleShapes<Self::Shape, RHS::Shape>>;
}

/// Union trait for types implementing `ZipFold`, `ZipMap` and `ZipMut`.
///
/// `ZipOps` is automatically implemented for all valid types and provides
/// a shortcut for general type constraints.
pub trait ZipOps<RHS: Buffer>:
    ZipFold<RHS>
    + ZipMap<RHS>
    + ZipMut<RHS>
where
    Self::Shape: Broadcast<RHS::Shape>,
{}

impl<A, B> ZipOps<B> for A
where
    A: ZipFold<B> + ZipMap<B> + ZipMut<B>,
    B: Buffer,

    A::Shape: Broadcast<B::Shape>,
{}

// ---------------------------------------------------------------------------
// Linear-Algebraic Operations
// ---------------------------------------------------------------------------
/// Trait for performing contractions over a pair of tensor buffers.
pub trait Contract<RHS: Buffer<Field = Self::Field>, const AXES: usize = 1>: Buffer {
    /// The post-contraction buffer type.
    type Output: Buffer<Field = Self::Field>;

    /// Return the contraction of two buffers.
    ///
    /// # Examples
    ///
    /// ```
    /// # use aegir::buffers::contract;
    /// let x = [1.0, 2.0, 3.0];
    /// let y = [-1.0, 0.0, 1.0];
    ///
    /// assert_eq!(contract::<1, _, _>(x, y).unwrap(), 2.0);
    /// ```
    fn contract(
        self,
        rhs: RHS,
    ) -> Result<Self::Output, IncompatibleShapes<Self::Shape, RHS::Shape>>;

    /// Return the contraction of two buffers specifications.
    ///
    /// This method can be much more efficient for large operations if you know that
    /// one side is, e.g., homogeneous or diagonal. As shown in the example below,
    /// we can exploit sparsity properties to reduce an O(n^3) operation into a trivial
    /// O(1) cacluation.
    ///
    /// # Examples
    ///
    /// ```
    /// # use aegir::buffers::{Spec, contract_spec, shapes::{S1, S2}};
    /// let x: Spec<[[f64; 3]; 3]> = Spec::Full(S2, 1.0);
    /// let y: Spec<[[f64; 3]; 3]> = Spec::Full(S2, 2.0);
    ///
    /// assert_eq!(contract_spec::<1, _, _>(x, y).unwrap(), Spec::Full(S2, 6.0));
    /// ```
    fn contract_spec(
        lhs: Spec<Self>,
        rhs: Spec<RHS>,
    ) -> Result<Spec<Self::Output>, IncompatibleShapes<Self::Shape, RHS::Shape>>;

    /// Return the shape of the contraction over two buffers.
    fn contract_shape(
        lhs: shapes::ShapeOf<Self>,
        rhs: shapes::ShapeOf<RHS>,
    ) -> Result<shapes::ShapeOf<Self::Output>, IncompatibleShapes<Self::Shape, RHS::Shape>>;
}

/// Return the contraction of two buffers.
pub fn contract<const AXES: usize, X, Y>(x: X, y: Y) -> Result<X::Output, IncompatibleShapes<X::Shape, Y::Shape>>
where
    X: Contract<Y, AXES>,
    Y: Buffer<Field = X::Field>,
{
    x.contract(y)
}

/// Return the contraction of two buffers specifications.
pub fn contract_spec<const AXES: usize, X, Y>(x: Spec<X>, y: Spec<Y>) -> Result<Spec<X::Output>, IncompatibleShapes<X::Shape, Y::Shape>>
where
    X: Contract<Y, AXES>,
    Y: Buffer<Field = X::Field>,
{
    <X as Contract<Y, AXES>>::contract_spec(x, y)
}

/// Return the shape of the contraction over two buffers.
pub fn contract_shape<const AXES: usize, X, Y>(x: X::Shape, y: Y::Shape) -> Result<shapes::ShapeOf<X::Output>, IncompatibleShapes<X::Shape, Y::Shape>>
where
    X: Contract<Y, AXES>,
    Y: Buffer<Field = X::Field>,
{
    <X as Contract<Y, AXES>>::contract_shape(x, y)
}

mod scalars;
pub use scalars::*;

mod tuples;
pub use tuples::*;

mod vecs;
pub use vecs::*;

mod arrays;
pub use arrays::*;

mod spec;
pub use spec::{IntoSpec, Spec};

// #[cfg(feature = "ndarray")]
// mod ndarray;
// #[cfg(feature = "ndarray")]
// pub use self::ndarray::*;
