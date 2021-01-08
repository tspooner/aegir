use crate::{
    Identifier, State, Contains, Function, Differentiable,
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

impl<T, N> Contains<T> for Sigmoid<N>
where
    T: Identifier,
    N: Contains<T>,
{
    fn contains(&self, target: T) -> bool { self.0.contains(target) }
}

impl<S, N> Function<S> for Sigmoid<N>
where
    S: State,
    N: Function<S>,

    <N::Codomain as Buffer>::Field: num_traits::real::Real,
{
    type Codomain = OwnedOf<N::Codomain>;
    type Error = N::Error;

    fn evaluate(&self, state: &S) -> Result<Self::Codomain, Self::Error> {
        self.0.evaluate(state).map(|buffer| buffer.map(sigmoid))
    }
}

impl<S, T, N> Differentiable<S, T> for Sigmoid<N>
where
    S: State,
    T: Identifier,
    N: Differentiable<S, T>,

    <N::Codomain as Buffer>::Field: Real,
    <N::Jacobian as Buffer>::Field: Real,
{
    type Jacobian = OwnedOf<N::Jacobian>;

    fn grad(&self, state: &S, target: T) -> Result<Self::Jacobian, Self::Error> {
        self.0.grad(state, target).map(|buffer| buffer.map(logistic))
    }
}

impl<N: fmt::Display> fmt::Display for Sigmoid<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "\u{03C3}{}", self.0)
    }
}
