use crate::{
    buffer::{Buffer, FieldOf},
    Compile,
    Contains,
    Database,
    Differentiable,
    Function,
    Identifier,
    Node,
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

impl<D, N> Function<D> for Reduce<N>
where
    D: Database,
    N: Function<D>,

    FieldOf<N::Codomain>: num_traits::Zero,
{
    type Codomain = FieldOf<N::Codomain>;
    type Error = N::Error;

    fn evaluate(&self, db: &D) -> Result<Self::Codomain, Self::Error> {
        self.0
            .evaluate(db)
            .map(|buffer| buffer.fold(num_traits::zero(), |acc, &x| acc + x))
    }
}

impl<D, T, N> Differentiable<D, T> for Reduce<N>
where
    D: Database,
    T: Identifier,
    N: Differentiable<D, T>,

    N::Jacobian: Buffer<Field = FieldOf<N::Codomain>>,

    FieldOf<N::Codomain>: num_traits::Zero,
{
    type Jacobian = N::Jacobian;

    fn grad(&self, db: &D, target: T) -> Result<Self::Jacobian, Self::Error> {
        self.0.grad(db, target)
    }
}

impl<T, N> Compile<T> for Reduce<N>
where
    T: Identifier,
    N: Compile<T>,
{
    type CompiledJacobian = N::CompiledJacobian;
    type Error = N::Error;

    fn compile_grad(&self, target: T) -> Result<Self::CompiledJacobian, Self::Error> {
        self.0.compile_grad(target)
    }
}

impl<N: fmt::Display> fmt::Display for Reduce<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "\u{03A3}_i ({})_i", self.0)
    }
}
