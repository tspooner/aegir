use crate::{
    ops::{AddOne, Div, Mul},
    Contains,
    Differentiable,
    Identifier,
    Node,
};
use num_traits::real::Real;

impl_unary!(
    /// Operator that applies `f[g](x) = ln(g(x))` element-wise to a buffer.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate aegir;
    /// # use aegir::{Identifier, Function, ids::X, ops::Ln};
    /// let op = Ln(X.into_var());
    ///
    /// assert!((op.evaluate(ctx!{X = 2.0f64.exp()}).unwrap() - 2.0).abs() < 1e-5);
    /// assert!((op.evaluate(ctx!{X = 4.0f64.exp()}).unwrap() - 4.0).abs() < 1e-5);
    /// ```
    Ln<F: Real>, |x| { x.ln() }, |self| {
        use crate::fmt::{PreWrap, Expr::*};

        match self.0.to_expr() {
            Zero => Text(PreWrap {
                text: "-\u{221E}".to_string(),
                needs_wrap: false,
            }),
            One => Zero,
            Text(pw) => Text(PreWrap {
                text: format!("ln({})", pw),
                needs_wrap: false,
            }),
        }
    }
);

impl<T, N> Differentiable<T> for Ln<N>
where
    T: Identifier,
    N: Differentiable<T> + Clone,
{
    type Adjoint = Div<N::Adjoint, N>;

    fn adjoint(&self, target: T) -> Self::Adjoint { Div(self.0.adjoint(target), self.0.clone()) }
}

impl_unary!(
    /// Operator that applies `f[g](x) = g(x) · ln(g(x))` element-wise to a buffer.
    ///
    /// This implementation is more numerically stable than the equivalent operator
    /// `Mul<Variable<X>, Ln<Variable<X>>>` and should be preferred where possible.
    /// The reason, is that `ln(0)` is not defined, but `0 · ln(0)` takes value zero.
    /// Note that this operator incurs some overhead due to conditional branching.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate aegir;
    /// # use aegir::{Identifier, Function, ids::X, ops::XLnX};
    /// let op = XLnX(X.into_var());
    ///
    /// assert!((op.evaluate(ctx!{X = 3.76868f64}).unwrap() - 5.0).abs() < 1e-5);
    /// ```
    XLnX<F: Real>, |x| {
        if x <= F::zero() { F::zero() } else { x * x.ln() }
    }, |self| {
        use crate::fmt::{PreWrap, Expr::*};

        match self.0.to_expr() {
            Zero => Zero,
            One => Zero,
            Text(pw) => Text(PreWrap {
                text: format!("{} \u{2218} ln({})", pw.to_safe_string('(', ')'), pw),
                needs_wrap: true,
            })
        }
    }
);

impl<T, N> Differentiable<T> for XLnX<N>
where
    T: Identifier,
    N: Differentiable<T> + Clone,
{
    type Adjoint = Mul<N::Adjoint, AddOne<Ln<N>>>;

    fn adjoint(&self, target: T) -> Self::Adjoint {
        self.0.adjoint(target).mul(AddOne(Ln(self.0.clone())))
    }
}

impl_unary!(
    /// Operator that applies `f[g](x) = exp(g(x))` element-wise to a buffer.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate aegir;
    /// # use aegir::{Identifier, Function, ids::X, ops::Exp};
    /// let op = Exp(X.into_var());
    ///
    /// assert!((op.evaluate(ctx!{X = 2.0f64.ln()}).unwrap() - 2.0).abs() < 1e-5);
    /// assert!((op.evaluate(ctx!{X = 4.0f64.ln()}).unwrap() - 4.0).abs() < 1e-5);
    /// ```
    Exp<F: Real>, |x| { x.exp() }, |self| {
        use crate::fmt::{PreWrap, Expr::*};

        match self.0.to_expr() {
            Zero => One,
            One => Text(PreWrap {
                text: "𝑒".to_string(),
                needs_wrap: false,
            }),
            Text(pw) => Text(PreWrap {
                text: format!("𝑒^{}", pw.to_safe_string('(', ')')),
                needs_wrap: false,
            }),
        }
    }
);

impl<T, N> Differentiable<T> for Exp<N>
where
    T: Identifier,
    N: Differentiable<T> + Clone,
{
    type Adjoint = Mul<Self, N::Adjoint>;

    fn adjoint(&self, target: T) -> Self::Adjoint { Mul(self.clone(), self.0.adjoint(target)) }
}
