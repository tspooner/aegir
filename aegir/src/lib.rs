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
//! - Decoupled/generic tensor type.
//! - Monadic runtime optimisation.
//! - Custom DSL for operator expansion.
#![deny(missing_docs)]

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

pub mod errors;
use errors::*;

/// Interface for type-level identifiers.
///
/// This trait should be implemented for symbols that are used to label variable/meta nodes (see
/// [meta]). For example, one might define `X` and `Y` for use in regression models, or `W` to
/// denote weights.  To make life easier, we define a large set of "standard" identifiers in the
/// [ids](ids/index.html) module.
///
/// Implementation of this trait is mostly uncomplicated, but can be cumbersome. In particular, the
/// [VariableAdjoint](meta::VariableAdjoint) type relies on `PartialEq` being implemented for the
/// two identifiers `I` and `T`. The procedural macro [ids!] is provided to make this simpler
/// should you want to define a custom type.
pub trait Identifier: Copy + PartialEq + Eq + std::fmt::Debug + std::fmt::Display {
    /// Convert the identifier into a [Variable].
    fn into_var(self) -> meta::Variable<Self> { meta::Variable(self) }
}

pub mod ids {
    //! Quality-of-life shortcuts for commonly-used identifiers.
    ids!(
        // Latin alphabet:
        A::a, B::b, C::c, D::d, E::e, F::f, G::g, H::h, I::i,
        J::j, K::k, L::l, M::m, N::n, O::o, P::p, Q::q, R::r,
        S::s, T::t, U::u, V::v, W::w, X::x, Y::y, Z::z,

        // Greek alphabet:
        Alpha::"\u{03B1}", Beta::"\u{03B2}", Gamma::"\u{03B3}", Delta::"\u{03B4}",
        Epsilon::"\u{03B5}", Zeta::"\u{03B6}", Eta::"\u{03B7}", Theta::"\u{03B8}",
        Iota::"\u{03B9}", Kappa::"\u{03BA}", Lambda::"\u{03BB}", Mu::"\u{03BC}",
        Nu::"\u{03BD}", Xi::"\u{03BE}", Omicron::"\u{03BF}", Pi::"\u{03C0}",
        Rho::"\u{03C1}", Sigma::"\u{03C2}", Tau::"\u{03C3}", Upsilon::"\u{03C4}",
        Phi::"\u{03C6}", Chi::"\u{03C7}", Psi::"\u{03C8}", Omega::"\u{03C9}"
    );
}

/// Trait for types that store data [buffers](buffers::Buffer).
pub trait Context: AsRef<Self> {}

/// Trait for reading entries out of a [Context].
pub trait Read<I: Identifier>: Context {
    /// The buffer type associated with the identifier `I`.
    type Buffer: buffers::Buffer;

    /// Returns a copy of the value associated with `ident`, if it exists.
    fn read(&self, ident: I) -> Option<Self::Buffer>;

    /// Returns a specification of the value associated with `ident`, if it exists.
    fn read_spec(&self, ident: I) -> Option<buffers::Spec<Self::Buffer>> {
        self.read(ident).map(buffers::Spec::Raw)
    }

    /// Returns the shape of the value associated with `ident`, if it exists.
    fn read_shape(&self, ident: I) -> Option<buffers::shapes::ShapeOf<Self::Buffer>> {
        use buffers::shapes::Shaped;

        self.read(ident).map(|buf| buf.shape())
    }
}

/// Helper macro for simple, auto-magical [Context] types.
#[macro_export]
macro_rules! ctx_type {
    ($name:ident { $($buf_name:ident: $buf_ident:ident),+ }) => {
        paste! {
            #[derive(Context)]
            pub struct $name<$([<__ $buf_ident>]),+> {
                $(#[id($buf_ident)] pub $buf_name: [<__ $buf_ident>]),+
            }
        }
    }
}

#[macro_export]
macro_rules! ctx {
    ($($key:ident = $value:expr),+) => {{
        paste! {
            ctx_type!(Ctx { $([<_ $key:lower>]: $key),+ });

            Ctx {
                $([<_ $key:lower>]: $value),+
            }
        }
    }}
}

/// Base trait for operator nodes.
pub trait Node {
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

    fn dot<N: Node>(self, other: N) -> ops::TensorDot<Self, N>
    where
        Self: Sized,
    {
        ops::Contract(self, other)
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

    fn ln(self) -> ops::Ln<Self>
    where
        Self: Sized,
    {
        ops::Ln(self)
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

    fn sigmoid(self) -> ops::Sigmoid<Self>
    where
        Self: Sized,
    {
        ops::Sigmoid(self)
    }
}

/// Trait for operator [Nodes](Node) that can assert their symbolic contents.
pub trait Contains<T: Identifier>: Node {
    /// Returns true if the identifier is present in the expression.
    fn contains(&self, ident: T) -> bool;
}

/// Trait for operator [Nodes](Node) that can be evaluated against a [Context].
pub trait Function<C: Context>: Node {
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
    /// assert_eq!(X.into_var().evaluate(ctx!{X = 1.0}).unwrap(), 1.0);
    /// ```
    fn evaluate<CR: AsRef<C>>(&self, ctx: CR) -> AegirResult<Self, C>;

    fn evaluate_spec<CR: AsRef<C>>(
        &self,
        ctx: CR,
    ) -> Result<buffers::Spec<Self::Value>, Self::Error> {
        self.evaluate(ctx).map(buffers::Spec::Raw)
    }

    /// Evaluate the function and return the shape of the
    /// [Value](Function::Value).
    ///
    /// __Note:__ by default, this method performs a full evaluation and calls
    /// the shape method on the buffer. This should be overridden in your
    /// implementation for better efficiency.
    fn evaluate_shape<CR: AsRef<C>>(
        &self,
        ctx: CR,
    ) -> Result<buffers::shapes::ShapeOf<Self::Value>, Self::Error> {
        self.evaluate(ctx)
            .map(|ref buf| buffers::shapes::Shaped::shape(buf))
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
    /// let c = 2.0f64.into_constant();
    /// let grad = X.into_var().mul(c).adjoint(X);
    ///
    /// assert_eq!(grad.evaluate(ctx!{X = 10.0}).unwrap(), 2.0);
    /// ```
    fn adjoint(&self, target: T) -> Self::Adjoint;

    /// Helper method that computes the adjoint and evaluates its value.
    ///
    /// __Note:__ this method can be more efficient than explicitly solving for
    /// the adjoint tree. In particular, this method can be implemented
    /// using direct numerical calculations.
    fn evaluate_adjoint<C: Context, CR: AsRef<C>>(
        &self,
        target: T,
        ctx: CR,
    ) -> AegirResult<Self::Adjoint, C>
    where
        Self: Function<C>,
        Self::Adjoint: Function<C>,
    {
        self.adjoint(target).evaluate(ctx)
    }

    /// Helper method that evaluates the function and its adjoint, wrapping up
    /// in a [Dual].
    fn evaluate_dual<C: Context, CR: AsRef<C>>(
        &self,
        target: T,
        ctx: CR,
    ) -> Result<
        DualOf<Self, C, T>,
        BinaryError<Self::Error, <AdjointOf<Self, T> as Function<C>>::Error, NoError>,
    >
    where
        Self: Function<C>,
        Self::Adjoint: Function<C>,
    {
        let value = self.evaluate(&ctx).map_err(BinaryError::Left)?;
        let adjoint = self.evaluate_adjoint(target, ctx).map_err(BinaryError::Right)?;

        Ok(dual!(value, adjoint))
    }
}

/// Alias for the error type associated with a function.
pub type ErrorOf<F, C> = <F as Function<C>>::Error;

/// Alias for the value type associated with a function.
pub type ValueOf<F, C> = <F as Function<C>>::Value;

/// Alias for the result type associated with a function.
pub type AegirResult<F, C> = Result<ValueOf<F, C>, ErrorOf<F, C>>;

/// Alias for the adjoint of a function.
pub type AdjointOf<F, T> = <F as Differentiable<T>>::Adjoint;

/// Alias for the dual associated with a function.
pub type DualOf<F, C, T> = Dual<ValueOf<F, C>, ValueOf<AdjointOf<F, T>, C>>;

extern crate self as aegir;

mod dual;
pub use self::dual::Dual;

pub mod fmt;
pub mod buffers;
pub mod meta;
pub mod ops;
