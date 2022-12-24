use aegir::{Node, Identifier, Differentiable};
use special_fun::FloatSpecial;

// Derive = x.gamma() * x.digamma()
impl_unary!(
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

impl_unary!(
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
    type Adjoint = crate::ops::Mul<N::Adjoint, DiGamma<N>>;

    fn adjoint(&self, target: I) -> Self::Adjoint {
        self.0.adjoint(target).mul(DiGamma(self.0.clone()))
    }
}

/// Operator alias that applies the factorial element-wise over a buffer.
pub type Factorial<N> = crate::ops::AddOne<Gamma<N>>;

impl<N> Factorial<N> {
    /// Create an instance of [Factorial].
    pub fn factorial(node: N) -> Self { crate::ops::AddOne(Gamma(node)) }
}

impl_unary!(
    /// Operator that applies `erf(.)` element-wise over a buffer.
    ///
    /// The error function is defined as 2 * Phi(sqrt(2) * x) - 1 where Phi(.) is the standard
    /// normal CDF.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate aegir;
    /// # use aegir::{Identifier, Function, ids::X, ops::Erf};
    /// ctx!(Ctx { x: X });
    ///
    /// let op = Erf(X.into_var());
    /// let opc = op.complement();
    ///
    /// assert_eq!(
    ///     op.evaluate(Ctx { x: 1.0f64, }).unwrap(),
    ///     1.0 - opc.evaluate(Ctx { x: 1.0f64, }).unwrap()
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
    /// Transform `erf(.)` operator into the complement `erfc(.)`.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate aegir;
    /// # use aegir::{Identifier, Function, ids::X, ops::Erf};
    /// ctx!(Ctx { x: X });
    ///
    /// let op = Erf(X.into_var());
    /// let opc = op.complement();
    ///
    /// assert_eq!(
    ///     op.evaluate(Ctx { x: 1.0f64, }).unwrap(),
    ///     1.0 - opc.evaluate(Ctx { x: 1.0f64, }).unwrap()
    /// );
    /// ```
    pub fn complement(self) -> crate::ops::OneSub<Self> { crate::ops::OneSub(self) }
}

/// Operator alias that applies `erfc(.)` element-wise over a buffer.
pub type Erfc<N> = crate::ops::OneSub<Erf<N>>;

impl<N> Erfc<N> {
    /// Create an instance of [Erfc].
    pub fn erfc(node: N) -> Erfc<N> { crate::ops::OneSub(Erf(node)) }
}
