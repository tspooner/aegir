use crate::{
    buffers::Buffer,
    Differentiable,
    Identifier,
};
use num_traits::real::Real;

impl_unary!(
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

fn sigmoid<F: Real>(x: F) -> F {
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
    /// Computes the element-wise sigmoid of a [Buffer].
    ///
    /// # Examples
    /// ```
    /// # #[macro_use] extern crate aegir;
    /// # use aegir::{Identifier, Differentiable, ops::Sigmoid, ids::X};
    /// db!(DB { x: X });
    ///
    /// let db = DB {
    ///     x: [1.0f64, 2.0f64, 3.0f64]
    /// };
    /// let dual = Sigmoid(X.into_var()).evaluate_dual(X, &db).unwrap();
    ///
    /// assert!((dual.value[0] - 0.73106).abs() < 1e-5);
    /// assert!((dual.value[1] - 0.88080).abs() < 1e-5);
    /// assert!((dual.value[2] - 0.95258).abs() < 1e-5);
    ///
    /// assert!((dual.adjoint[0] - 0.19661).abs() < 1e-5);
    /// assert!((dual.adjoint[1] - 0.10499).abs() < 1e-5);
    /// assert!((dual.adjoint[2] - 0.04518).abs() < 1e-5);
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
    type Adjoint = crate::ops::TensorDot<N::Adjoint, Rabbit<Self>>;

    fn adjoint(&self, target: T) -> Self::Adjoint {
        crate::ops::TensorDot::new(self.0.adjoint(target), Rabbit(self.clone()))
    }
}
