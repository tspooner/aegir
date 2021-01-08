use crate::{
    Identifier, State, Node, Contains, Function, Differentiable,
    buffer::{Buffer, FieldOf},
};
use std::fmt;

#[derive(Copy, Clone, Debug)]
pub struct Reduce<N>(pub N);

impl<N> Node for Reduce<N> {}

impl<T, N> Contains<T> for Reduce<N>
where
    T: Identifier,
    N: Contains<T>,
{
    fn contains(&self, target: T) -> bool { self.0.contains(target) }
}

impl<S, N> Function<S> for Reduce<N>
where
    S: State,
    N: Function<S>,
{
    type Codomain = FieldOf<N::Codomain>;
    type Error = N::Error;

    fn evaluate(&self, state: &S) -> Result<Self::Codomain, Self::Error> {
        self.0.evaluate(state).map(|buffer| {
            buffer.fold(num_traits::zero(), |acc, &x| acc + x)
        })
    }
}

impl<S, T, N> Differentiable<S, T> for Reduce<N>
where
    S: State,
    T: Identifier,
    N: Differentiable<S, T>,

    N::Jacobian: Buffer<Field = FieldOf<N::Codomain>>,
{
    type Jacobian = N::Jacobian;

    fn grad(&self, state: &S, target: T) -> Result<Self::Jacobian, Self::Error> {
        self.0.grad(state, target)
    }
}

impl<N: fmt::Display> fmt::Display for Reduce<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "\u{03A3}_i {}_i", self.0)
    }
}
