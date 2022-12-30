use crate::{
    buffers::{Buffer, Scalar, Spec, shapes::{S0, Shape, ShapeOf}},
    Contains,
    Context,
    Differentiable,
    Function,
    Identifier,
    Node,
};
use num_traits::{real::Real, FromPrimitive};
use std::fmt;

impl_unary!(
    /// Operator that applies `f[g](x) = g(x) - 1` element-wise to a buffer.
    ///
    /// # Examples
    /// ```
    /// # #[macro_use] extern crate aegir;
    /// # use aegir::{Identifier, Differentiable, Dual, ops::SubOne, ids::X};
    /// let f = SubOne(X.into_var());
    ///
    /// assert_eq!(f.evaluate_dual(X, ctx!{X = 1.0}).unwrap(), dual!(0.0, 1.0));
    /// assert_eq!(f.evaluate_dual(X, ctx!{X = 0.0}).unwrap(), dual!(-1.0, 1.0));
    /// assert_eq!(f.evaluate_dual(X, ctx!{X = -1.0}).unwrap(), dual!(-2.0, 1.0));
    /// ```
    SubOne<F: Scalar>, |x| { x - F::one() }, |self| {
        use crate::fmt::{PreWrap, Expr::*};

        match self.0.to_expr() {
            Zero => Text(PreWrap {
                text: "-1".to_string(),
                needs_wrap: false,
            }),
            One => Zero,
            Text(pw) => Text(PreWrap {
                text: format!("{} - 1", pw.to_safe_string('(', ')')),
                needs_wrap: true,
            })
        }
    }
);

impl<T, N> Differentiable<T> for SubOne<N>
where
    T: Identifier,
    N: Differentiable<T>,
{
    type Adjoint = N::Adjoint;

    fn adjoint(&self, target: T) -> Self::Adjoint { self.0.adjoint(target) }
}

impl_unary!(
    /// Operator that applies `f[g](x) = 1 - g(x)` element-wise to a buffer.
    ///
    /// # Examples
    /// ```
    /// # #[macro_use] extern crate aegir;
    /// # use aegir::{Identifier, Differentiable, Dual, ops::OneSub, ids::X};
    /// let f = OneSub(X.into_var());
    ///
    /// assert_eq!(f.evaluate_dual(X, ctx!{X = 1.0}).unwrap(), dual!(0.0, -1.0));
    /// assert_eq!(f.evaluate_dual(X, ctx!{X = 0.0}).unwrap(), dual!(1.0, -1.0));
    /// assert_eq!(f.evaluate_dual(X, ctx!{X = -1.0}).unwrap(), dual!(2.0, -1.0));
    /// ```
    OneSub<F: Scalar>, |x| { F::one() - x }, |self| {
        use crate::fmt::{PreWrap, Expr::*};

        match self.0.to_expr() {
            Zero => One,
            One => Zero,
            Text(pw) => Text(PreWrap {
                text: format!("1 - {}", pw.to_safe_string('(', ')')),
                needs_wrap: true,
            })
        }
    }
);

impl<T, N> Differentiable<T> for OneSub<N>
where
    T: Identifier,
    N: Differentiable<T>,
{
    type Adjoint = Negate<N::Adjoint>;

    fn adjoint(&self, target: T) -> Self::Adjoint { Negate(self.0.adjoint(target)) }
}

impl_unary!(
    /// Operator that applies `f[g](x) = g(x) + 1` element-wise to a buffer.
    ///
    /// # Examples
    /// ```
    /// # #[macro_use] extern crate aegir;
    /// # use aegir::{Identifier, Differentiable, Dual, ops::AddOne, ids::X};
    /// let f = AddOne(X.into_var());
    ///
    /// assert_eq!(f.evaluate_dual(X, ctx!{X = 1.0}).unwrap(), dual!(2.0, 1.0));
    /// assert_eq!(f.evaluate_dual(X, ctx!{X = 0.0}).unwrap(), dual!(1.0, 1.0));
    /// assert_eq!(f.evaluate_dual(X, ctx!{X = -1.0}).unwrap(), dual!(0.0, 1.0));
    /// ```
    AddOne<F: Scalar>, |x| { x + F::one() }, |self| {
        use crate::fmt::{PreWrap, Expr::*};

        match self.0.to_expr() {
            Zero => One,
            One => Text(PreWrap {
                text: "2".to_string(),
                needs_wrap: true,
            }),
            Text(pw) => Text(PreWrap {
                text: format!("1 - {}", pw.to_safe_string('(', ')')),
                needs_wrap: true,
            })
        }
    }
);

impl<T, N> Differentiable<T> for AddOne<N>
where
    T: Identifier,
    N: Differentiable<T>,
{
    type Adjoint = N::Adjoint;

    fn adjoint(&self, target: T) -> Self::Adjoint { self.0.adjoint(target) }
}

impl_unary!(
    /// Operator that applies `f[g](x) = g(x) ^ 2` element-wise to a buffer.
    ///
    /// # Examples
    /// ```
    /// # #[macro_use] extern crate aegir;
    /// # use aegir::{Identifier, Differentiable, Dual, ops::Square, ids::X};
    /// let f = Square(X.into_var());
    ///
    /// assert_eq!(f.evaluate_dual(X, ctx!{X = -1.0}).unwrap(), dual!(1.0, -2.0));
    /// assert_eq!(f.evaluate_dual(X, ctx!{X = 0.0}).unwrap(), dual!(0.0, 0.0));
    /// assert_eq!(f.evaluate_dual(X, ctx!{X = 1.0}).unwrap(), dual!(1.0, 2.0));
    /// assert_eq!(f.evaluate_dual(X, ctx!{X = 2.0}).unwrap(), dual!(4.0, 4.0));
    /// ```
    Square<F: num_traits::Pow<F, Output = F>>, |x| { x.pow(F::one() + F::one()) }, |self| {
        use crate::fmt::{PreWrap, Expr::*};

        match self.0.to_expr() {
            Zero => Zero,
            One => One,
            Text(pw) => Text(PreWrap {
                text: format!("({})^2", pw.to_safe_string('(', ')')),
                needs_wrap: false,
            })
        }
    }
);

impl<T, N> Differentiable<T> for Square<N>
where
    T: Identifier,
    N: Differentiable<T> + Clone,
{
    type Adjoint = crate::ops::Mul<Double<N>, N::Adjoint>;

    fn adjoint(&self, target: T) -> Self::Adjoint {
        crate::ops::Mul(Double(self.0.clone()), self.0.adjoint(target))
    }
}

impl_unary!(
    /// Operator that applies `f[g](x) = 2 · g(x)` element-wise to a buffer.
    ///
    /// # Examples
    /// ```
    /// # #[macro_use] extern crate aegir;
    /// # use aegir::{Identifier, Differentiable, Dual, ops::Double, ids::X};
    /// let f = Double(X.into_var());
    ///
    /// assert_eq!(f.evaluate_dual(X, ctx!{X = -1.0}).unwrap(), dual!(-2.0, 2.0));
    /// assert_eq!(f.evaluate_dual(X, ctx!{X = 0.0}).unwrap(), dual!(0.0, 2.0));
    /// assert_eq!(f.evaluate_dual(X, ctx!{X = 1.0}).unwrap(), dual!(2.0, 2.0));
    /// ```
    Double<F: Scalar>, |x| { (F::one() + F::one()) * x }, |self| {
        use crate::fmt::{PreWrap, Expr::*};

        match self.0.to_expr() {
            Zero => Zero,
            One => Text(PreWrap {
                text: "2".to_string(),
                needs_wrap: false,
            }),
            Text(pw) => Text(PreWrap {
                text: format!("2 \u{2218} {}", pw.to_safe_string('(', ')')),
                needs_wrap: false,
            })
        }
    }
);

impl<T, N> Differentiable<T> for Double<N>
where
    T: Identifier,
    N: Differentiable<T>,
{
    type Adjoint = Double<N::Adjoint>;

    fn adjoint(&self, target: T) -> Self::Adjoint { Double(self.0.adjoint(target)) }
}

/// Operator that applies `f[g](x) = Σᵢ gᵢ(x)` to a buffer.
///
/// # Examples
/// ```
/// # #[macro_use] extern crate aegir;
/// # use aegir::{Identifier, Differentiable, Dual, ops::Sum, ids::X};
/// let f = Sum(X.into_var());
///
/// assert_eq!(
///     f.evaluate_dual(X, ctx!{X = [1.0, 2.0, 3.0]}).unwrap(),
///     dual!(6.0, [
///         [1.0, 0.0, 0.0],
///         [0.0, 1.0, 0.0],
///         [0.0, 0.0, 1.0]
///     ])
/// );
/// ```
#[derive(Copy, Clone, Debug, PartialEq, Contains)]
pub struct Sum<N>(#[op] pub N);

impl<N: Node> Node for Sum<N> {}

impl<C, N, F> Function<C> for Sum<N>
where
    C: Context,
    N: Function<C>,
    F: Scalar + FromPrimitive,

    N::Value: Buffer<Field = F>,
{
    type Error = N::Error;
    type Value = F;

    fn evaluate<CR: AsMut<C>>(&self, ctx: CR) -> Result<Self::Value, Self::Error> {
        self.0.evaluate(ctx).map(|buf| buf.sum())
    }

    fn evaluate_spec<CR: AsMut<C>>(&self, ctx: CR) -> Result<Spec<Self::Value>, Self::Error> {
        use Spec::*;

        Ok(match self.0.evaluate_spec(ctx)? {
            // TODO: replace unwrap with error propagation.
            Full(sh, val) => Full(S0, F::from_usize(sh.cardinality()).unwrap() * val),
            spec => Raw(spec.unwrap().sum()),
        })
    }

    #[inline]
    fn evaluate_shape<CR: AsMut<C>>(&self, _: CR) -> Result<ShapeOf<Self::Value>, Self::Error> {
        Ok(S0)
    }
}

impl<T, N> Differentiable<T> for Sum<N>
where
    T: Identifier,
    N: Differentiable<T>,
{
    type Adjoint = N::Adjoint;

    fn adjoint(&self, target: T) -> Self::Adjoint { self.0.adjoint(target) }
}

impl<N: fmt::Display> fmt::Display for Sum<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "\u{03A3}_i ({})_i", self.0)
    }
}

impl_unary!(
    /// Operator that applies `f[g](x) = -g(x)` to a buffer.
    ///
    /// # Examples
    /// ```
    /// # #[macro_use] extern crate aegir;
    /// # use aegir::{Identifier, Differentiable, Dual, ops::Negate, ids::X};
    /// let f = Negate(X.into_var());
    ///
    /// assert_eq!(f.evaluate_dual(X, ctx!{X = 1.0}).unwrap(), dual!(-1.0, -1.0));
    /// assert_eq!(f.evaluate_dual(X, ctx!{X = -1.0}).unwrap(), dual!(1.0, -1.0));
    /// ```
    Negate<F: num_traits::real::Real>, |x| { -x }, |self| {
        use crate::fmt::{PreWrap, Expr::*};

        match self.0.to_expr() {
            Zero => Zero,
            One => Text(PreWrap {
                text: "-1".to_string(),
                needs_wrap: false,
            }),
            Text(pw) => Text(PreWrap {
                text: format!("-{}", pw.to_safe_string('(', ')')),
                needs_wrap: false,
            })
        }
    }
);

impl<T, N> Differentiable<T> for Negate<N>
where
    T: Identifier,
    N: Differentiable<T>,
{
    type Adjoint = Negate<N::Adjoint>;

    fn adjoint(&self, target: T) -> Self::Adjoint { Negate(self.0.adjoint(target)) }
}

impl_unary!(
    /// Operator that applies `f[g](x) = δ(g(x) - x)` to a buffer.
    ///
    /// # Examples
    /// ```
    /// # #[macro_use] extern crate aegir;
    /// # use aegir::{Identifier, Function, Dual, ops::Dirac, ids::X};
    /// # use std::f64::INFINITY;
    /// let f = Dirac(X.into_var());
    ///
    /// assert_eq!(f.evaluate(ctx!{X = -1.0}).unwrap(), 0.0);
    /// assert_eq!(f.evaluate(ctx!{X = 0.0}).unwrap(), INFINITY);
    /// assert_eq!(f.evaluate(ctx!{X = 1.0}).unwrap(), 0.0);
    /// ```
    Dirac<F: num_traits::Float>, |x| {
        match x {
            _ if (x == num_traits::zero()) => num_traits::Float::infinity(),
            _ => num_traits::zero(),
        }
    }, |self| {
        use crate::fmt::{PreWrap, Expr::*};

        Text(PreWrap {
            text: format!("\u{03B4}({})", self.0.to_expr()),
            needs_wrap: false,
        })
    }
);

impl_unary!(
    /// Operator that applies `f[g](x) = sgn(g(x))` to a buffer.
    ///
    /// # Examples
    /// ```
    /// # #[macro_use] extern crate aegir;
    /// # use aegir::{Identifier, Differentiable, Dual, ops::Sign, ids::X};
    /// # use std::f64::INFINITY;
    /// let f = Sign(X.into_var());
    ///
    /// assert_eq!(f.evaluate_dual(X, ctx!{X = -1.0}).unwrap(), dual!(-1.0, 0.0));
    /// assert_eq!(f.evaluate_dual(X, ctx!{X = 0.0}).unwrap(), dual!(0.0, INFINITY));
    /// assert_eq!(f.evaluate_dual(X, ctx!{X = 1.0}).unwrap(), dual!(1.0, 0.0));
    /// ```
    Sign<F: num_traits::Float>, |x| {
        if num_traits::Zero::is_zero(&x) { x } else { x.signum() }
    }, |self| {
        use crate::fmt::{PreWrap, Expr::*};

        Text(PreWrap {
            text: format!("sgn({})", self.0.to_expr()),
            needs_wrap: false,
        })
    }
);

impl<T, N> Differentiable<T> for Sign<N>
where
    T: Identifier,
    N: Differentiable<T> + Clone,
{
    type Adjoint = crate::ops::Mul<Dirac<N>, N::Adjoint>;

    fn adjoint(&self, target: T) -> Self::Adjoint {
        Dirac(self.0.clone()).mul(self.0.adjoint(target))
    }
}

impl_unary!(
    /// Operator that applies `f[g](x) = |g(x)|` to a buffer.
    ///
    /// # Examples
    /// ```
    /// # #[macro_use] extern crate aegir;
    /// # use aegir::{Identifier, Differentiable, Dual, ops::Abs, ids::X};
    /// let f = Abs(X.into_var());
    ///
    /// assert_eq!(f.evaluate_dual(X, ctx!{X = -1.0}).unwrap(), dual!(1.0, -1.0));
    /// assert_eq!(f.evaluate_dual(X, ctx!{X = 0.0}).unwrap(), dual!(0.0, 0.0));
    /// assert_eq!(f.evaluate_dual(X, ctx!{X = 1.0}).unwrap(), dual!(1.0, 1.0));
    /// ```
    Abs<F: Real>, |x| { x.abs() }, |self| {
        use crate::fmt::{PreWrap, Expr::*};

        match self.0.to_expr() {
            Zero => Zero,
            One => One,
            Text(pw) => Text(PreWrap {
                text: format!("|{}|", pw),
                needs_wrap: false,
            })
        }
    }
);

impl<T, N> Differentiable<T> for Abs<N>
where
    T: Identifier,
    N: Differentiable<T> + Clone,
{
    type Adjoint = crate::ops::Mul<Sign<N>, N::Adjoint>;

    fn adjoint(&self, target: T) -> Self::Adjoint {
        Sign(self.0.clone()).mul(self.0.adjoint(target))
    }
}
