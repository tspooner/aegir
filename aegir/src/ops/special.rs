//! Module containing special function operators.
use aegir::{Node, Identifier, Differentiable, ops::{Mul, OneSub, AddOne}};
use special_fun::FloatSpecial;

impl_unary!(
    /// Operator that applies `f[g](x) = Γ(f(x)) = (f(x) - 1)!` element-wise to a buffer.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate aegir;
    /// # use aegir::{Identifier, Function, ids::X, ops::special::Gamma};
    /// let op = Gamma(X.into_var());
    ///
    /// assert_eq!(op.evaluate(ctx!{X = 2.0f64}).unwrap(), 1.0);
    /// assert_eq!(op.evaluate(ctx!{X = 3.0f64}).unwrap(), 2.0);
    /// assert_eq!(op.evaluate(ctx!{X = 4.0f64}).unwrap(), 6.0);
    /// ```
    Gamma<F: FloatSpecial>, |x| { x.gamma() }, |self| {
        use crate::fmt::{PreWrap, Expr::*};

        match self.0.to_expr() {
            Zero => Text(PreWrap {
                text: "\u{221E}".to_string(),
                needs_wrap: false,
            }),
            One => One,
            Text(pw) => Text(PreWrap {
                text: format!("\u{0393}({})", pw),
                needs_wrap: false,
            })
        }
    }
);

impl<N, I> Differentiable<I> for Gamma<N>
where
    N: Clone + Differentiable<I>,
    I: Identifier,
{
    type Adjoint = Mul<Mul<N::Adjoint, DiGamma<N>>, Gamma<N>>;

    fn adjoint(&self, target: I) -> Self::Adjoint {
        self.0.adjoint(target).mul(DiGamma(self.0.clone())).mul(self.clone())
    }
}

impl_unary!(
    /// Operator that applies `f[g](x) = ψ(g(x)) = Γ'(g(x)) / Γ(g(x))` element-wise to a buffer.
    DiGamma<F: FloatSpecial>, |x| { x.digamma() }, |self| {
        use crate::fmt::{PreWrap, Expr::*};

        match self.0.to_expr() {
            Zero => Text(PreWrap {
                text: "\u{221E}".to_string(),
                needs_wrap: false,
            }),
            One => One,
            Text(pw) => Text(PreWrap {
                text: format!("\u{03C8}({})", pw),
                needs_wrap: false,
            })
        }
    }
);

impl_unary!(
    /// Operator that applies `f[g](x) = ln Γ(g(x))` element-wise to a buffer.
    LogGamma<F: FloatSpecial>, |x| { x.loggamma() }, |self| {
        use crate::fmt::{PreWrap, Expr::*};

        match self.0.to_expr() {
            Zero => Text(PreWrap {
                text: "\u{221E}".to_string(),
                needs_wrap: false,
            }),
            One => Zero,
            Text(pw) => Text(PreWrap {
                text: format!("ln\u{0393}({})", pw),
                needs_wrap: false,
            })
        }
    }
);

impl<N, I> Differentiable<I> for LogGamma<N>
where
    N: Clone + Differentiable<I>,
    I: Identifier,
{
    type Adjoint = Mul<N::Adjoint, DiGamma<N>>;

    fn adjoint(&self, target: I) -> Self::Adjoint {
        self.0.adjoint(target).mul(DiGamma(self.0.clone()))
    }
}

/// Operator alias that applies `f[g](x) = g(x)!` element-wise to a buffer.
///
/// # Examples
///
/// ```
/// # #[macro_use] extern crate aegir;
/// # use aegir::{Identifier, Function, ids::X, ops::special::Factorial};
/// let op = Factorial::factorial(X.into_var());
///
/// assert_eq!(op.evaluate(ctx!{X = 1.0f64}).unwrap(), 1.0);
/// assert_eq!(op.evaluate(ctx!{X = 5.0f64}).unwrap(), 120.0);
/// ```
pub type Factorial<N> = Gamma<AddOne<N>>;

impl<N> Factorial<N> {
    /// Create an instance of [Factorial].
    pub fn factorial(node: N) -> Self { Gamma(AddOne(node)) }
}

impl_unary!(
    /// Operator that applies `f[g](x) = erf(g(x))` element-wise to a buffer.
    ///
    /// The error function is defined as 2 * Phi(sqrt(2) * x) - 1 where Phi(.) is the standard
    /// normal CDF.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate aegir;
    /// # use aegir::{Identifier, Function, ids::X, ops::special::Erf};
    /// let op = Erf(X.into_var());
    /// let opc = op.complement();
    ///
    /// assert_eq!(
    ///     op.evaluate(ctx!{X = 1.0f64}).unwrap(),
    ///     1.0 - opc.evaluate(ctx!{X = 1.0f64}).unwrap()
    /// );
    /// ```
    Erf<F: FloatSpecial>, |x| { x.erf() }, |self| {
        use crate::fmt::{PreWrap, Expr::*};

        match self.0.to_expr() {
            Zero | One => One,
            Text(pw) => Text(PreWrap {
                text: format!("erf({})", pw),
                needs_wrap: false,
            })
        }
    }
);

impl<N> Erf<N> {
    /// Transform operator into its complement `Erfc`.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate aegir;
    /// # use aegir::{Identifier, Function, ids::X, ops::special::Erf};
    /// let op = Erf(X.into_var());
    /// let opc = op.complement();
    ///
    /// assert_eq!(
    ///     op.evaluate(ctx!{X = 1.0f64}).unwrap(),
    ///     1.0 - opc.evaluate(ctx!{X = 1.0f64}).unwrap()
    /// );
    /// ```
    pub fn complement(self) -> OneSub<Self> { OneSub(self) }
}

/// Operator alias that applies `f[g](x) = erfc(g(x))` element-wise to a buffer.
pub type Erfc<N> = OneSub<Erf<N>>;

impl<N> Erfc<N> {
    /// Create an instance of [Erfc].
    pub fn erfc(node: N) -> Erfc<N> { OneSub(Erf(node)) }
}
