//! Module containing trigonometric operators.
use crate::{
    ops::{Mul, Negate},
    Differentiable,
    Identifier,
};
use num_traits::real::Real;

impl_unary!(
    /// Operator that applies `f[g](x) = cos(g(x))` element-wise to a buffer.
    Cos<F: Real>, |x| { x.cos() }, |self| {
        use crate::fmt::{PreWrap, Expr::*};

        match self.0.to_expr() {
            Zero => One,
            One => Text(PreWrap {
                text: "cos(1)".to_string(),
                needs_wrap: false,
            }),
            Text(pw) => Text(PreWrap {
                text: format!("cos({})", pw),
                needs_wrap: false,
            })
        }
    }
);

impl<T, N> Differentiable<T> for Cos<N>
where
    T: Identifier,
    N: Differentiable<T> + Clone,
{
    type Adjoint = Negate<Mul<N::Adjoint, Sin<N>>>;

    fn adjoint(&self, ident: T) -> Self::Adjoint {
        Negate(Mul(self.0.adjoint(ident), Sin(self.0.clone())))
    }
}

impl_unary!(
    /// Operator that applies `f[g](x) = cosh(g(x))` element-wise to a buffer.
    Cosh<F: Real>, |x| { x.cosh() }, |self| {
        use crate::fmt::{PreWrap, Expr::*};

        match self.0.to_expr() {
            Zero => One,
            One => Text(PreWrap {
                text: "cosh(1)".to_string(),
                needs_wrap: false,
            }),
            Text(pw) => Text(PreWrap {
                text: format!("cosh({})", pw),
                needs_wrap: false,
            })
        }
    }
);

impl_unary!(
    /// Operator that applies `f[g](x) = acos(g(x))` element-wise to a buffer.
    ArcCos<F: Real>, |x| { x.acos() }, |self| {
        use crate::fmt::{PreWrap, Expr::*};

        match self.0.to_expr() {
            Zero => Text(PreWrap {
                text: "\u{03C0}/2".to_string(),
                needs_wrap: false,
            }),
            One => Zero,
            Text(pw) => Text(PreWrap {
                text: format!("acos({})", pw),
                needs_wrap: false,
            })
        }
    }
);

impl_unary!(
    /// Operator that applies `f[g](x) = acosh(g(x))` element-wise to a buffer.
    ArcCosh<F: Real>, |x| { x.acosh() }, |self| {
        use crate::fmt::{PreWrap, Expr::*};

        match self.0.to_expr() {
            Zero => Text(PreWrap {
                text: "\u{1D456}\u{03C0}/2".to_string(),
                needs_wrap: false,
            }),
            One => Zero,
            Text(pw) => Text(PreWrap {
                text: format!("acosh({})", pw),
                needs_wrap: false,
            })
        }
    }
);

impl_unary!(
    /// Operator that applies `f[g](x) = sin(g(x))` element-wise to a buffer.
    Sin<F: Real>, |x| { x.sin() }, |self| {
        use crate::fmt::{PreWrap, Expr::*};

        match self.0.to_expr() {
            Zero => Zero,
            One => Text(PreWrap {
                text: "sin(1)".to_string(),
                needs_wrap: false,
            }),
            Text(pw) => Text(PreWrap {
                text: format!("sin({})", pw),
                needs_wrap: false,
            })
        }
    }
);

impl<T, N> Differentiable<T> for Sin<N>
where
    T: Identifier,
    N: Differentiable<T> + Clone,
{
    type Adjoint = Mul<N::Adjoint, Cos<N>>;

    fn adjoint(&self, ident: T) -> Self::Adjoint {
        Mul(self.0.adjoint(ident), Cos(self.0.clone()))
    }
}

impl_unary!(
    /// Operator that applies `f[g](x) = sinh(g(x))` element-wise to a buffer.
    Sinh<F: Real>, |x| { x.sinh() }, |self| {
        use crate::fmt::{PreWrap, Expr::*};

        match self.0.to_expr() {
            Zero => Zero,
            One => Text(PreWrap {
                text: "sin(1)".to_string(),
                needs_wrap: false,
            }),
            Text(pw) => Text(PreWrap {
                text: format!("sinh({})", pw),
                needs_wrap: false,
            })
        }
    }
);

impl_unary!(
    /// Operator that applies `f[g](x) = asin(g(x))` element-wise to a buffer.
    ArcSin<F: Real>, |x| { x.asin() }, |self| {
        use crate::fmt::{PreWrap, Expr::*};

        match self.0.to_expr() {
            Zero => Zero,
            One => Text(PreWrap {
                text: "\u{03C0}/2".to_string(),
                needs_wrap: false,
            }),
            Text(pw) => Text(PreWrap {
                text: format!("acos({})", pw),
                needs_wrap: false,
            })
        }
    }
);

impl_unary!(
    /// Operator that applies `f[g](x) = asinh(g(x))` element-wise to a buffer.
    ArcSinh<F: Real>, |x| { x.asinh() }, |self| {
        use crate::fmt::{PreWrap, Expr::*};

        match self.0.to_expr() {
            Zero => Zero,
            One => Text(PreWrap {
                text: "asinh(1)".to_string(),
                needs_wrap: false,
            }),
            Text(pw) => Text(PreWrap {
                text: format!("asinh({})", pw),
                needs_wrap: false,
            })
        }
    }
);

impl_unary!(
    /// Operator that applies `f[g](x) = tan(g(x))` element-wise to a buffer.
    Tan<F: Real>, |x| { x.tan() }, |self| {
        use crate::fmt::{PreWrap, Expr::*};

        match self.0.to_expr() {
            Zero => Zero,
            One => Text(PreWrap {
                text: "tan(1)".to_string(),
                needs_wrap: false,
            }),
            Text(pw) => Text(PreWrap {
                text: format!("tan({})", pw),
                needs_wrap: false,
            })
        }
    }
);

impl_unary!(
    /// Operator that applies `f[g](x) = tanh(g(x))` element-wise to a buffer.
    Tanh<F: Real>, |x| { x.tanh() }, |self| {
        use crate::fmt::{PreWrap, Expr::*};

        match self.0.to_expr() {
            Zero => Zero,
            One => Text(PreWrap {
                text: "tanh(1)".to_string(),
                needs_wrap: false,
            }),
            Text(pw) => Text(PreWrap {
                text: format!("tanh({})", pw),
                needs_wrap: false,
            })
        }
    }
);

impl_unary!(
    /// Operator that applies `f[g](x) = atan(g(x))` element-wise to a buffer.
    ArcTan<F: Real>, |x| { x.atan() }, |self| {
        use crate::fmt::{PreWrap, Expr::*};

        match self.0.to_expr() {
            Zero => Zero,
            One => Text(PreWrap {
                text: "\u{03C0}/4".to_string(),
                needs_wrap: false,
            }),
            Text(pw) => Text(PreWrap {
                text: format!("atan({})", pw),
                needs_wrap: false,
            })
        }
    }
);

impl_unary!(
    /// Operator that applies `f[g](x) = atanh(g(x))` element-wise to a buffer.
    ArcTanh<F: Real>, |x| { x.atanh() }, |self| {
        use crate::fmt::{PreWrap, Expr::*};

        match self.0.to_expr() {
            Zero => Zero,
            One => Text(PreWrap {
                text: "\u{221E}".to_string(),
                needs_wrap: false,
            }),
            Text(pw) => Text(PreWrap {
                text: format!("atanh({})", pw),
                needs_wrap: false,
            })
        }
    }
);
