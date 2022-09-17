use crate::{
    buffers::{Buffer, OwnedOf},
    Contains,
    Differentiable,
    Function,
    Read,
    Identifier,
    Node,
};

#[derive(PartialEq)]
pub struct NamedNode<I, N>(pub I, pub N);

impl<I, N> Node for NamedNode<I, N> {}

impl<T, I, N> Contains<T> for NamedNode<I, N>
where
    T: Identifier,
    I: Identifier + std::cmp::PartialEq<T>,
    N: Contains<T>,
{
    fn contains(&self, target: T) -> bool { self.0 == target || self.1.contains(target) }
}

impl<D, N, I> Function<D> for NamedNode<N, I>
where
    D: Read<I>,
    N: Function<D, Value = OwnedOf<D::Buffer>>,
    I: Identifier,
{
    type Error = N::Error;
    type Value = N::Value;

    fn evaluate<DR: AsRef<D>>(&self, db: DR) -> Result<Self::Value, Self::Error> {
        if let Some(v) = db.as_ref().read(self.1) {
            Ok(v.to_owned())
        } else {
            self.0.evaluate(db)
        }
    }
}

impl<N, I, T> Differentiable<T> for NamedNode<N, I>
where
    N: Differentiable<T>,
    I: Identifier + std::cmp::PartialEq<T>,
    T: Identifier,
{
    type Adjoint = N::Adjoint;

    fn adjoint(&self, target: T) -> Self::Adjoint { self.0.adjoint(target) }
}
