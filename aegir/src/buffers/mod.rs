//! Module for tensor abstractions and their implementations.
pub mod shapes;

use shapes::{Concat, Ix, Shape};

/// Marker trait for class subscriptions of [Buffer] types.
///
/// This trait is used to define a semantic grouping over buffer types.
/// For a given class, implementing this trait defines a unique
/// mapping between a [Shape](Buffer::Shape) and [Field](Buffer::Field),
/// and a (sized) [Buffer] type.
///
/// # Examples
/// The Class trait is particularly useful for constructing instances of
/// buffer types. In the example below, we consider the [Arrays] class
/// which subsumes all compile-time homoegeneous data buffers. Note that we
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
pub trait Class<S: Shape, F: Scalar> {
    /// The associated buffer types.
    type Buffer: Buffer<Class = Self, Shape = S, Field = F> + Sized;

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
    fn build(shape: S, f: impl Fn(S::Index) -> F) -> Self::Buffer;

    /// Construct a [Buffer](Class::Buffer) with default value and subset
    /// assignment.
    ///
    /// This method is generally more efficient than [Class::build] as the scan
    /// is only performed over the subset of entries provided.
    ///
    /// # Examples
    /// ```
    /// # #[macro_use] extern crate aegir;
    /// # use aegir::buffers::{Class, Arrays, shapes::{S2, Indices}};
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
    fn build_subset(
        shape: S,
        base: F,
        subset: impl Iterator<Item = shapes::IndexOf<S>>,
        active: impl Fn(S::Index) -> F,
    ) -> Self::Buffer;

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
    /// # use aegir::buffers::{Class, Tuples, shapes::{S1, Indices}};
    /// assert_eq!(Tuples::full(S1, 0.0), (0.0, 0.0));
    /// assert_eq!(Tuples::full(S1, 1.0), (1.0, 1.0));
    /// ```
    fn full(shape: S, value: F) -> Self::Buffer { Self::build(shape, |_| value) }

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
    /// override; this is usually very easy if `S` implements
    /// [Indices](shapes::Indices).
    ///
    /// # Examples
    /// ```
    /// # #[macro_use] extern crate aegir;
    /// # use aegir::buffers::{Class, Arrays, shapes::{S2, Indices}};
    /// assert_eq!(Arrays::diagonal(S2, 5.0), [
    ///     [5.0, 0.0],
    ///     [0.0, 5.0]
    /// ]);
    /// ```
    fn diagonal(shape: S, value: F) -> Self::Buffer {
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
    /// # use aegir::buffers::{Class, Arrays, shapes::{S2, Indices}};
    /// let eye: [[f64; 2]; 2] = Arrays::identity(S2);
    ///
    /// assert_eq!(eye, [
    ///     [1.0, 0.0],
    ///     [0.0, 1.0]
    /// ]);
    /// ```
    fn identity(shape: S) -> Self::Buffer { Self::diagonal(shape, num_traits::one()) }
}

/// Trait for types defining a data buffer over a fixed [Field](Buffer::Field)
/// and [Shape](Buffer::Shape).
pub trait Buffer: std::fmt::Debug {
    /// [Class] associated with the buffer.
    type Class: Class<Self::Shape, Self::Field>;

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
    fn map<F>(self, f: F) -> OwnedOf<Self>
    where
        F: Fn(Self::Field) -> Self::Field;

    /// Perform an element-wise transformation of the buffer (reference).
    fn map_ref<F>(&self, f: F) -> OwnedOf<Self>
    where
        F: Fn(Self::Field) -> Self::Field;

    /// Perform a fold over the elements of the buffer.
    ///
    /// # Examples
    /// ```
    /// # use aegir::buffers::Buffer;
    /// let buffer = vec![0.0, 1.0, 2.0, 3.0];
    ///
    /// assert_eq!(buffer.fold(0.0, |init, &el| init + 2.0 * el), 12.0);
    /// ```
    fn fold<F>(&self, init: Self::Field, f: F) -> Self::Field
    where
        F: Fn(Self::Field, &Self::Field) -> Self::Field;

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
    fn sum(&self) -> Self::Field
    where
        Self::Field: num_traits::Zero,
    {
        self.fold(num_traits::zero(), |init, &el| init + el)
    }

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

    /// Create an owned buffer of all zeroes with a given shape.
    fn zeroes(shape: Self::Shape) -> OwnedOf<Self> {
        Self::Class::full(shape, num_traits::identities::zero())
    }

    /// Create an owned buffer of all zeroes with the same shape as self.
    fn to_zeroes(&self) -> OwnedOf<Self> { self.to_filled(num_traits::identities::zero()) }

    /// Replace the contents of a buffer with zeroes.
    fn into_zeroes(self) -> OwnedOf<Self>
    where
        Self: Sized,
    {
        self.into_filled(num_traits::identities::zero())
    }

    /// Create an owned buffer of all ones with a given shape.
    fn ones(shape: Self::Shape) -> OwnedOf<Self> {
        Self::Class::full(shape, num_traits::identities::one())
    }

    /// Create an owned buffer of all ones with the same shape as self.
    fn to_ones(&self) -> OwnedOf<Self> { self.to_filled(num_traits::identities::one()) }

    /// Replace the contents of a buffer with ones.
    fn into_ones(self) -> OwnedOf<Self>
    where
        Self: Sized,
    {
        self.into_filled(num_traits::identities::one())
    }

    /// Create an owned buffer of a given value with the same shape as self.
    fn to_filled(&self, value: Self::Field) -> OwnedOf<Self> { self.to_owned().map(|_| value) }

    /// Replace the contents of a buffer with a given value.
    fn into_filled(self, value: Self::Field) -> OwnedOf<Self>
    where
        Self: Sized,
    {
        self.map(|_| value)
    }
}

/// Utility class for implementations of [PrecedenceMapping].
pub struct Precedence;

/// Trait that maps
pub trait PrecedenceMapping<F, S1, C1, S2, C2>
where
    F: Scalar,
    S1: Concat<S2>,
    C1: Class<S1, F>,
    S2: Shape,
    C2: Class<S2, F>,
{
    type Class: Class<<S1 as Concat<S2>>::Shape, F>;
}

pub type PrecedenceOf<F, S1, C1, S2, C2> =
    <Precedence as PrecedenceMapping<F, S1, C1, S2, C2>>::Class;

macro_rules! impl_class_precedence {
    (($cl1:ty, $cl2:ty) => $cl3:ty) => {
        impl<F, S1, S2> PrecedenceMapping<F, S1, $cl1, S2, $cl2> for Precedence
        where
            F: Scalar,
            S1: Concat<S2>,
            S2: Shape,

            $cl1: Class<S1, F>,
            $cl2: Class<S2, F>,
            $cl3: Class<<S1 as Concat<S2>>::Shape, F>,
        {
            type Class = $cl3;
        }
    };
}

impl_class_precedence!((Scalars, Scalars) => Scalars);
impl_class_precedence!((Scalars, Arrays) => Arrays);
impl_class_precedence!((Scalars, Tuples) => Tuples);
impl_class_precedence!((Scalars, Vecs) => Vecs);

impl_class_precedence!((Arrays, Scalars) => Arrays);
impl_class_precedence!((Arrays, Arrays) => Arrays);
impl_class_precedence!((Arrays, Tuples) => Arrays);
impl_class_precedence!((Arrays, Vecs) => Arrays);

impl_class_precedence!((Tuples, Scalars) => Tuples);
impl_class_precedence!((Tuples, Arrays) => Arrays);
impl_class_precedence!((Tuples, Tuples) => Tuples);
impl_class_precedence!((Tuples, Vecs) => Vecs);

impl_class_precedence!((Vecs, Scalars) => Vecs);
impl_class_precedence!((Vecs, Arrays) => Vecs);
impl_class_precedence!((Vecs, Tuples) => Vecs);
impl_class_precedence!((Vecs, Vecs) => Vecs);

/// Type shortcut for the [Class] associated with a [Buffer].
pub type ClassOf<B> = <B as Buffer>::Class;

/// Type shortcut for the [Shape] associated with a [Buffer].
pub type ShapeOf<B> = <B as Buffer>::Shape;

/// Type shortcut for the [Field] associated with a [Buffer].
pub type FieldOf<B> = <B as Buffer>::Field;

/// Type shortcut for the owned variant of a [Buffer].
pub type OwnedOf<B> =
    <<B as Buffer>::Class as Class<<B as Buffer>::Shape, <B as Buffer>::Field>>::Buffer;

#[derive(Debug)]
pub struct IncompatibleBuffers<D1, D2 = D1>(pub(crate) D1, pub(crate) D2);

impl<D1, D2> std::fmt::Display for IncompatibleBuffers<D1, D2>
where
    D1: std::fmt::Display,
    D2: std::fmt::Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Buffers are incompatible: {} vs {}.", self.0, self.1)
    }
}

impl<D: std::fmt::Debug + std::fmt::Display> std::error::Error for IncompatibleBuffers<D> {}

pub trait Coalesce<RHS = Self>: Buffer
where
    RHS: Buffer<Field = Self::Field>,
{
    fn coalesce(
        &self,
        rhs: &RHS,
        init: Self::Field,
        f: impl Fn(Self::Field, (Self::Field, Self::Field)) -> Self::Field,
    ) -> Result<Self::Field, IncompatibleBuffers<Self::Shape, RHS::Shape>>;
}

pub trait Hadamard<RHS = Self>: Buffer
where
    RHS: Buffer<Field = Self::Field>,
{
    type Output: Buffer<Field = Self::Field>;

    fn hadamard(
        &self,
        rhs: &RHS,
        f: impl Fn(Self::Field, Self::Field) -> Self::Field,
    ) -> Result<Self::Output, IncompatibleBuffers<Self::Shape, RHS::Shape>>;
}

pub trait Compatible<B: Buffer>: Buffer<Field = FieldOf<B>> + Coalesce<B> + Hadamard<B> {}

impl<B, T> Compatible<B> for T
where
    B: Buffer,
    T: Buffer<Field = FieldOf<B>> + Coalesce<B> + Hadamard<B>,
{
}

mod scalars;
pub use scalars::*;

mod tuples;
pub use tuples::*;

mod vecs;
pub use vecs::*;

mod arrays;
pub use arrays::*;

#[cfg(feature = "ndarray")]
mod ndarray;
#[cfg(feature = "ndarray")]
pub use self::ndarray::*;
