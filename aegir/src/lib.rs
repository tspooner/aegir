//! Strongly-typed, compile-time autodifferentiation in Rust.
//!
//! `aegir` is an experimental autodifferentiation framework designed to
//! leverage the powerful type-system in Rust and _avoid runtime as much as
//! humanly possible_. The approach taken resembles that of expression
//! templates, as commonly used in linear-algebra libraries written in C++, and
//! indeed mirrors (in many ways) the concept of [Iterator] in [std].
//!
//! A key distinction of `aegir` from existing autodiff frameworks is that it
//! does not rely on a monolithic `Tensor` type that handles shape
//! transformations at runtime. This would be equivalent to using the ndarray
//! crate to handle all numerical computations and data-storage/layouts. While
//! this has its advantages - like simplicity and ease of use - our approach is
//! a strict generalisation and allows for much greater flexibility. By
//! incorporating both dynamic and fixed-size data structures, we also
//! have the advantage of the various compile-time optimisations that come from
//! using arrays, tuples, and scalars.
//!
//! # Key Features
//! - Built-in arithmetic, linear-algebraic, trigonometric and special
//!   operators.
//! - Infinitely differentiable: _Jacobian, Hessian, etc..._
//! - Custom DSL for operator expansion.
//! - Decoupled/generic tensor type.
// //! - Plug-and-play backends.
#[allow(unused_imports)]
#[macro_use]
extern crate aegir_derive;
#[doc(hidden)]
pub use self::aegir_derive::*;

#[allow(unused_imports)]
#[macro_use]
extern crate aegir_compile;
#[doc(hidden)]
pub use self::aegir_compile::*;

#[allow(unused_imports)]
use paste::paste;

#[macro_use]
extern crate itertools;

/// Placeholder for infallible operations.
///
/// This type is mostly used in conjunction with [BinaryError] for operators
/// that guarantee no errors within their own transformation code. This is
/// better than relying on `either` as the semantic meaning of the three error
/// cases is preserved.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum NoError {}

impl std::fmt::Display for NoError {
    fn fmt(&self, _: &mut std::fmt::Formatter) -> std::fmt::Result { match *self {} }
}

impl std::error::Error for NoError {
    fn description(&self) -> &str { match *self {} }
}

/// Error type for unary operations.
///
/// This type is typically used when an operator is defined over a single
/// sub-operator; e.g. negation or trigonometric functions. The
/// [Inner](UnaryError::Inner) case allows for error propagation from the
/// underlying operator, and unique errors arising within the given
/// operator itself via [Output](UnaryError::Output).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum UnaryError<I, O> {
    /// Inner operator failed.
    Inner(I),

    /// Error occurred post-hoc.
    Output(O),
}

impl<I, O> std::fmt::Display for UnaryError<I, O>
where
    I: std::fmt::Display,
    O: std::fmt::Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            UnaryError::Inner(e) => e.fmt(f),
            UnaryError::Output(e) => e.fmt(f),
        }
    }
}

impl<I, O> std::error::Error for UnaryError<I, O>
where
    I: std::fmt::Debug + std::fmt::Display,
    O: std::fmt::Debug + std::fmt::Display,
{
}

/// Error type for binary operations.
///
/// This type is typically used when an operator is defined over a pair of
/// sub-operators; e.g. addition, multiplication, subtraction. The
/// [Left](BinaryError::Left) and [Right](BinaryError::Right) cases then allow
/// for error propagation from said pair, as well as unique errors arising
/// within the given operator itself via [Output](BinaryError::Output).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum BinaryError<L, R, O> {
    /// Left operator failed.
    Left(L),

    /// Right operator failed.
    Right(R),

    /// Error occurred post-hoc.
    Output(O),
}

impl<L, R, O> std::fmt::Display for BinaryError<L, R, O>
where
    L: std::fmt::Display,
    R: std::fmt::Display,
    O: std::fmt::Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BinaryError::Left(e) => e.fmt(f),
            BinaryError::Right(e) => e.fmt(f),
            BinaryError::Output(e) => e.fmt(f),
        }
    }
}

impl<L, R, O> std::error::Error for BinaryError<L, R, O>
where
    L: std::fmt::Debug + std::fmt::Display,
    R: std::fmt::Debug + std::fmt::Display,
    O: std::fmt::Debug + std::fmt::Display,
{
}

/// Trait for type-level identifiers.
///
/// This trait should be implemented for every symbol that is to be used in the
/// computation. For example, one might define identifiers `X` and `Y` for use
/// in regression models. Implementation, however, is mostly hidden to the
/// regular user and one doesn't typically implement this manually, but rather
/// uses the procedural macro [ids!].
///
/// Note: some quality-of-life shortcuts are provided in the
/// [ids](ids/index.html) module.
pub trait Identifier: Eq + Copy + PartialEq + std::fmt::Debug + std::fmt::Display {
    /// Convert the identifier into a [Variable].
    fn into_var(self) -> Variable<Self> { Variable(self) }
}

pub mod ids {
    //! Quality-of-life shortcuts for commonly-used identifiers.
    ids!(
        A::a,
        B::b,
        C::c,
        D::d,
        E::e,
        F::f,
        G::g,
        H::h,
        I::i,
        J::j,
        K::k,
        L::l,
        M::m,
        N::n,
        O::o,
        P::p,
        Q::q,
        R::r,
        S::s,
        T::t,
        U::u,
        V::v,
        W::w,
        X::x,
        Y::y,
        Z::z
    );

    ids!(
        Alpha::"\u{03B1}", Beta::"\u{03B2}", Gamma::"\u{03B3}", Delta::"\u{03B4}",
        Epsilon::"\u{03B5}", Zeta::"\u{03B6}", Eta::"\u{03B7}", Theta::"\u{03B8}",
        Iota::"\u{03B9}", Kappa::"\u{03BA}", Lambda::"\u{03BB}", Mu::"\u{03BC}",
        Nu::"\u{03BD}", Xi::"\u{03BE}", Omicron::"\u{03BF}", Pi::"\u{03C0}",
        Rho::"\u{03C1}", Sigma::"\u{03C2}", Tau::"\u{03C3}", Upsilon::"\u{03C4}",
        Phi::"\u{03C6}", Chi::"\u{03C7}", Psi::"\u{03C8}", Omega::"\u{03C9}"
    );
}

/// Trait for types that store data [buffers](buffers::Buffer).
pub trait Database: AsRef<Self> {}

/// Trait for reading entries out of a [Database].
pub trait Read<I: Identifier>: Database {
    type Buffer: buffers::Buffer;

    fn read(&self, ident: I) -> Option<&Self::Buffer>;

    fn read_shape(&self, ident: I) -> Option<buffers::ShapeOf<Self::Buffer>> {
        use buffers::Buffer;

        self.read(ident).map(|buf| buf.shape())
    }
}

/// Helper macro for simple, auto-magical [Database] types.
#[macro_export]
macro_rules! db {
    ($name:ident { $($entity_name:ident: $entity_type:ident),+ }) => {
        paste! {
            #[derive(Database)]
            pub struct $name<$([<__ $entity_type>]),+> {
                $(#[id($entity_type)] pub $entity_name: [<__ $entity_type>]),+
            }
        }
    }
}

#[derive(Clone, Copy)]
pub enum Stage<N> {
    Type,
    Instance(N),
    Evaluation(N),
}

impl<N> Stage<N> {
    pub fn map<U>(self, f: impl Fn(N) -> U) -> Stage<U> {
        match self {
            Stage::Type => Stage::Type,
            Stage::Instance(node) => Stage::Instance(f(node)),
            Stage::Evaluation(node) => Stage::Evaluation(f(node)),
        }
    }

    pub fn split<O>(self, f_ct: impl Fn() -> O, f_rt: impl Fn(N) -> O) -> O {
        match self {
            Stage::Type => f_ct(),
            Stage::Instance(node) | Stage::Evaluation(node) => f_rt(node),
        }
    }
}

impl<'a, N: Node> Stage<&'a N> {
    pub fn is_zero(self) -> logic::TFU { N::is_zero(self) }

    pub fn is_one(self) -> logic::TFU { N::is_one(self) }
}

/// Base trait for operator nodes.
pub trait Node {
    // Optimizer functions:
    /// Checks if the node is zero-valued at a given stage.
    ///
    /// Note: this function returns a ternary truth value; i.e. it is possible
    /// that the answer is unknown at the function call-site.
    fn is_zero(_: Stage<&'_ Self>) -> logic::TFU { logic::TFU::Unknown }

    /// Checks if the node is unity-valued at a given stage.
    ///
    /// Note: this function returns a ternary truth value; i.e. it is possible
    /// that the answer is unknown at the function call-site.
    fn is_one(_: Stage<&'_ Self>) -> logic::TFU { logic::TFU::Unknown }

    // Combinator functions:
    fn add<N: Node>(self, other: N) -> ops::Add<Self, N>
    where
        Self: Sized,
    {
        ops::Add(self, other)
    }

    fn sub<N: Node>(self, other: N) -> ops::Sub<Self, N>
    where
        Self: Sized,
    {
        ops::Sub(self, other)
    }

    fn mul<N: Node>(self, other: N) -> ops::Mul<Self, N>
    where
        Self: Sized,
    {
        ops::Mul(self, other)
    }

    fn div<N: Node>(self, other: N) -> ops::Div<Self, N>
    where
        Self: Sized,
    {
        ops::Div(self, other)
    }

    fn dot<N: Node>(self, other: N) -> ops::InnerProduct<Self, N>
    where
        Self: Sized,
    {
        ops::InnerProduct(self, other)
    }

    fn tensor_mul<N: Node>(self, other: N) -> ops::TensorMul<Self, N>
    where
        Self: Sized,
    {
        ops::TensorMul(self, other)
    }

    fn abs(self) -> ops::Abs<Self>
    where
        Self: Sized,
    {
        ops::Abs(self)
    }

    fn neg(self) -> ops::Negate<Self>
    where
        Self: Sized,
    {
        ops::Negate(self)
    }

    fn pow<P>(self, power: P) -> ops::Power<Self, P>
    where
        Self: Sized,
    {
        ops::Power(self, power)
    }

    fn squared(self) -> ops::Square<Self>
    where
        Self: Sized,
    {
        ops::Square(self)
    }

    fn sum(self) -> ops::Sum<Self>
    where
        Self: Sized,
    {
        ops::Sum(self)
    }
}

/// Trait for operator [Nodes](Node) that can assert their symbolic contents.
pub trait Contains<T: Identifier>: Node {
    /// Returns true if the identifier is present in the expression.
    fn contains(&self, ident: T) -> bool;
}

/// Trait for operator [Nodes](Node) that can be evaluated against a [Database].
pub trait Function<D: Database>: Node {
    /// The codomain of the function.
    type Value: buffers::Buffer;

    /// The error type of the function.
    type Error: std::error::Error;

    /// Evaluate the function and return the corresponding
    /// [Value](Function::Value).
    ///
    /// # Examples
    /// ```
    /// # #[macro_use] extern crate aegir;
    /// # use aegir::{Identifier, Function, ids::X};
    /// db!(DB { x: X });
    ///
    /// assert_eq!(X.into_var().evaluate(DB { x: 1.0 }).unwrap(), 1.0);
    /// ```
    fn evaluate<DR: AsRef<D>>(&self, db: DR) -> AegirResult<Self, D>;

    /// Evaluate the function and return the shape of the
    /// [Value](Function::Value).
    ///
    /// __Note:__ by default, this method performs a full evaluation and calls
    /// the shape method on the buffer. This should be overridden in your
    /// implementation for better efficiency.
    fn evaluate_shape<DR: AsRef<D>>(
        &self,
        db: DR,
    ) -> Result<buffers::ShapeOf<Self::Value>, Self::Error> {
        self.evaluate(db).map(|ref buf| buffers::Buffer::shape(buf))
    }
}

/// Trait for operator [Nodes](Node) with a well-defined adjoint.
pub trait Differentiable<T: Identifier>: Node {
    /// The adjoint operator; i.e. the gradient.
    type Adjoint: Node;

    /// Transform the node into its [Adjoint](Differentiable::Adjoint) operator
    /// tree.
    ///
    /// This is the key method used to perform differentiation in `aegir`. For a
    /// given node, the derivative can be found by first computing the
    /// adjoint tree and then evaluating against a database as per
    /// [Function].
    ///
    /// # Examples
    /// ```
    /// # #[macro_use] extern crate aegir;
    /// # use aegir::{Node, Identifier, Differentiable, Function, buffers::Buffer, ids::X};
    /// db!(DB { x: X });
    ///
    /// let c = 2.0f64.into_constant();
    /// let grad = X.into_var().mul(c).adjoint(X);
    ///
    /// assert_eq!(grad.evaluate(DB { x: 10.0 }).unwrap(), 2.0);
    /// ```
    fn adjoint(&self, target: T) -> Self::Adjoint;

    /// Helper method that computes the adjoint and evaluates its value.
    ///
    /// __Note:__ this method can be more efficient than explicitly solving for
    /// the adjoint tree. In particular, this method can be implemented
    /// using direct numerical calculations.
    fn evaluate_adjoint<D: Database, DR: AsRef<D>>(
        &self,
        target: T,
        db: DR,
    ) -> AegirResult<Self::Adjoint, D>
    where
        Self::Adjoint: Function<D>,
    {
        self.adjoint(target).evaluate(db)
    }

    /// Helper method that evaluates the function and its adjoint, wrapping up
    /// in a [Dual].
    fn evaluate_dual<D: Database, DR: AsRef<D>>(
        &self,
        target: T,
        db: DR,
    ) -> Result<
        DualOf<Self, D, T>,
        BinaryError<Self::Error, <AdjointOf<Self, T> as Function<D>>::Error, NoError>,
    >
    where
        Self: Function<D>,
        Self::Adjoint: Function<D>,
    {
        let value = self.evaluate(db.as_ref()).map_err(BinaryError::Left)?;
        let adjoint = self
            .adjoint(target)
            .evaluate(db)
            .map_err(BinaryError::Right)?;

        Ok(dual!(value, adjoint))
    }
}

/// Alias for the error type associated with a function.
pub type ErrorOf<F, D> = <F as Function<D>>::Error;

/// Alias for the value type associated with a function.
pub type ValueOf<F, D> = <F as Function<D>>::Value;

/// Alias for the result type associated with a function.
pub type AegirResult<F, D> = Result<ValueOf<F, D>, ErrorOf<F, D>>;

/// Alias for the adjoint of a function.
pub type AdjointOf<F, T> = <F as Differentiable<T>>::Adjoint;

/// Alias for the dual associated with a function.
pub type DualOf<F, D, T> = Dual<ValueOf<F, D>, ValueOf<AdjointOf<F, T>, D>>;

extern crate self as aegir;

#[macro_use]
mod macros;

mod dual;
pub use self::dual::Dual;

mod sources;
pub use self::sources::{Constant, ConstantAdjoint, Variable, VariableAdjoint, VariableError};

pub mod buffers;
pub mod logic;
pub mod ops;

// mod named;
// pub use self::named::NamedNode;
