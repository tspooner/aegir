use crate::{
    Function, Differentiable, Node, Identifier,
    buffer::{Buffer, FieldOf},
};
use std::fmt;

#[derive(Copy, Clone, Debug)]
pub struct Reduce<N>(pub N);

impl<N> Node for Reduce<N> {}

impl<S, N> Function<S> for Reduce<N>
where
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

impl<T, S, N> Differentiable<T, S> for Reduce<N>
where
    T: Identifier,
    N: Differentiable<T, S>,

    N::Jacobian: Buffer<Field = FieldOf<N::Codomain>>,
{
    type Jacobian = N::Jacobian;

    fn grad(&self, target: T, state: &S) -> Result<Self::Jacobian, Self::Error> {
        self.0.grad(target, state)
    }
}

impl<N: fmt::Display> fmt::Display for Reduce<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "\u{03A3}_i {}_i", self.0)
    }
}
