use crate::{
    Identifier, Database, Contains, Function, Differentiable,
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

#[derive(Clone, Copy, Debug, Node, Contains)]
pub struct Sigmoid<N>(#[op] pub N);

impl<D, N> Function<D> for Sigmoid<N>
where
    D: Database,
    N: Function<D>,

    <N::Codomain as Buffer>::Field: num_traits::real::Real,
{
    type Codomain = OwnedOf<N::Codomain>;
    type Error = N::Error;

    fn evaluate(&self, db: &D) -> Result<Self::Codomain, Self::Error> {
        self.0.evaluate(db).map(|buffer| buffer.map(sigmoid))
    }
}

impl<D, T, N> Differentiable<D, T> for Sigmoid<N>
where
    D: Database,
    T: Identifier,
    N: Differentiable<D, T>,

    <N::Codomain as Buffer>::Field: Real,
    <N::Jacobian as Buffer>::Field: Real,
{
    type Jacobian = OwnedOf<N::Jacobian>;

    fn grad(&self, db: &D, target: T) -> Result<Self::Jacobian, Self::Error> {
        self.0.grad(db, target).map(|buffer| buffer.map(logistic))
    }
}

impl<N: fmt::Display> fmt::Display for Sigmoid<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "\u{03C3}{}", self.0)
    }
}
