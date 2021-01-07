use crate::{
    Function, Differentiable, Node, Identifier,
    buffer::{Buffer, OwnedOf},
};
use num_traits::{one, zero, real::Real};
use std::fmt;

fn sigmoid<F: Real>(x: F) -> F {
    if x >= zero() {
        let l: F = one();

        l / (l + (-x).exp())
    } else {
        let l: F = one();
        let z = x.exp();

        return z / (l + z)
    }
}

fn logistic<F: Real>(x: F) -> F {
    let l: F = one();

    l / (l + (-x).exp())
}

new_op!(Sigmoid<N>);

impl<S, N> Function<S> for Sigmoid<N>
where
    N: Function<S>,
    <N::Codomain as Buffer>::Field: num_traits::real::Real,
{
    type Codomain = OwnedOf<N::Codomain>;
    type Error = N::Error;

    fn evaluate(&self, state: &S) -> Result<Self::Codomain, Self::Error> {
        self.0.evaluate(state).map(|buffer| buffer.map(sigmoid))
    }
}

impl<T, S, N> Differentiable<T, S> for Sigmoid<N>
where
    T: Identifier,
    N: Differentiable<T, S>,
    <N::Codomain as Buffer>::Field: Real,
    <N::Jacobian as Buffer>::Field: Real,
{
    type Jacobian = OwnedOf<N::Jacobian>;

    fn grad(&self, target: T, state: &S) -> Result<Self::Jacobian, Self::Error> {
        self.0.grad(target, state).map(|buffer| buffer.map(logistic))
    }
}

impl<N: fmt::Display> fmt::Display for Sigmoid<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "\u{03C3}{}", self.0)
    }
}
