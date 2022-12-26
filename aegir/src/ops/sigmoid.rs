use crate::{Node, Differentiable, Identifier, ops::{OneSub, Double, Mul}};
use num_traits::real::Real;

impl_unary!(
    /// Operator that applies `f[g](x) = g(x) * (1 - g(x))` element-wise to a buffer.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate aegir;
    /// # use aegir::{Identifier, Differentiable, ops::Rabbit, ids::X};
    /// let op = Rabbit(X.into_var());
    /// let dual = op.evaluate_dual(X, ctx!{X = 0.5f64}).unwrap();
    ///
    /// assert!((dual.value - 0.25).abs() < 1e-5);
    /// assert!((dual.adjoint - 0.0).abs() < 1e-5);
    /// ```
    Rabbit<F: crate::buffers::Scalar>, |x| { x * (F::one() - x) }, |self| {
        use crate::fmt::{PreWrap, Expr::*};

        match self.0.to_expr() {
            Zero | One => Zero,
            Text(pw) => Text(PreWrap {
                text: format!("{0} \u{2218} (1 - {0})", pw.to_safe_string('(', ')')),
                needs_wrap: true,
            })
        }
    }
);

impl<N, I> Differentiable<I> for Rabbit<N>
where
    N: Clone + Differentiable<I>,
    I: Identifier,
{
    type Adjoint = Mul<OneSub<Double<N>>, N::Adjoint>;

    fn adjoint(&self, target: I) -> Self::Adjoint {
        OneSub(Double(self.0.clone())).mul(self.0.adjoint(target))
    }
}

/// Apply the sigmoid function to a real scalar value.
pub fn sigmoid<F: Real>(x: F) -> F {
    if x >= num_traits::zero() {
        let l: F = num_traits::one();

        l / (l + (-x).exp())
    } else {
        let l: F = num_traits::one();
        let z = x.exp();

        return z / (l + z);
    }
}

impl_unary!(
    /// Operator that applies `f[g](x) = 1 / (1 + exp(-f(x)))` element-wise to a buffer.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[macro_use] extern crate aegir;
    /// # use aegir::{Identifier, Differentiable, ops::Sigmoid, ids::X};
    /// let op = Sigmoid(X.into_var());
    /// let dual = op.evaluate_dual(X, ctx!{X = 1.0f64}).unwrap();
    ///
    /// assert!((dual.value - 0.73106).abs() < 1e-5);
    /// assert!((dual.adjoint - 0.19661).abs() < 1e-5);
    /// ```
    Sigmoid<F: Real>, |x| { sigmoid(x) }, |self| {
        use crate::fmt::{PreWrap, Expr::*};

        match self.0.to_expr() {
            Zero => Text(PreWrap {
                text: "\u{00BD}".to_string(),
                needs_wrap: false,
            }),
            One => Text(PreWrap {
                text: "\u{03C3}(1)".to_string(),
                needs_wrap: false,
            }),
            Text(pw) => Text(PreWrap {
                text: format!("\u{03C3}({})", pw),
                needs_wrap: false,
            })
        }
    }
);

impl<T, N> Differentiable<T> for Sigmoid<N>
where
    T: Identifier,
    N: Differentiable<T> + Clone,
{
    type Adjoint = crate::ops::Mul<N::Adjoint, Rabbit<Self>>;

    fn adjoint(&self, target: T) -> Self::Adjoint {
        crate::ops::Mul(self.0.adjoint(target), Rabbit(self.clone()))
    }
}
