use crate::{
    Identifier, Database, Get, Node, Contains, Function, Differentiable,
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

impl<D, N, I> Function<D> for NamedNode<N, I>
where
    D: Database + Get<I>,
    N: Function<D, Codomain = OwnedOf<D::Output>>,
    I: Identifier,

    D::Output: Buffer,
{
    type Codomain = N::Codomain;
    type Error = N::Error;

    fn evaluate(&self, state: &D) -> Result<Self::Codomain, Self::Error> {
        if let Some(v) = state.get(self.1) {
            Ok(v.to_owned())
        } else {
            self.0.evaluate(state)
        }
    }
}

impl<D, T, N, I> Differentiable<D, T> for NamedNode<N, I>
where
    D: Database + Get<I>,
    T: Identifier,
    N: Differentiable<D, T, Codomain = OwnedOf<D::Output>>,
    I: Identifier + std::cmp::PartialEq<T>,

    D::Output: Buffer,
{
    type Jacobian = N::Jacobian;

    fn grad(&self, state: &D, target: T) -> Result<Self::Jacobian, Self::Error> {
        self.0.grad(state, target)
    }
}
