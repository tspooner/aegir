use crate::{
    ops::{AddOne, Div, Mul},
    Contains,
    Differentiable,
    Identifier,
    Node,
};
use num_traits::real::Real;

impl_unary!(
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
    SafeXlnX<F: Real>, |x| {
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

impl<T, N> Differentiable<T> for SafeXlnX<N>
where
    T: Identifier,
    N: Differentiable<T> + Clone,
{
    type Adjoint = Mul<N::Adjoint, AddOne<Ln<N>>>;

    fn adjoint(&self, target: T) -> Self::Adjoint {
        self.0.adjoint(target).mul(AddOne(Ln(self.0.clone())))
    }
}
