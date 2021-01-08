use crate::{
    Identifier, State, Get, Node, Contains, Function, Differentiable,
    buffer::{Buffer, OwnedOf},
};

pub struct NamedNode<N, I>(pub N, pub I);

impl<N, I> Node for NamedNode<N, I> {}

impl<T, N, I> Contains<T> for NamedNode<N, I>
where
    T: Identifier,
    N: Contains<T>,
    I: Identifier + std::cmp::PartialEq<T>,
{
    fn contains(&self, target: T) -> bool {
        self.1 == target || self.0.contains(target)
    }
}

impl<S, N, I> Function<S> for NamedNode<N, I>
where
    S: State + Get<I>,
    N: Function<S, Codomain = OwnedOf<S::Output>>,
    I: Identifier,

    S::Output: Buffer,
{
    type Codomain = N::Codomain;
    type Error = N::Error;

    fn evaluate(&self, state: &S) -> Result<Self::Codomain, Self::Error> {
        if let Some(v) = state.get(self.1) {
            Ok(v.to_owned())
        } else {
            self.0.evaluate(state)
        }
    }
}

impl<S, T, N, I> Differentiable<S, T> for NamedNode<N, I>
where
    S: State + Get<I>,
    T: Identifier,
    N: Differentiable<S, T, Codomain = OwnedOf<S::Output>>,
    I: Identifier + std::cmp::PartialEq<T>,

    S::Output: Buffer,
{
    type Jacobian = N::Jacobian;

    fn grad(&self, state: &S, target: T) -> Result<Self::Jacobian, Self::Error> {
        self.0.grad(state, target)
    }
}
