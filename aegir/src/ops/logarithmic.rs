use crate::{
    buffers::{OwnedOf, Scalar},
    ops::{AddOne, Div, Mul},
    Contains,
    Database,
    Differentiable,
    Function,
    Identifier,
    Node,
};
use std::fmt;

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Ln<N>(pub N);

impl<N> Node for Ln<N> {}

impl<T, N> Contains<T> for Ln<N>
where
    T: Identifier,
    N: Contains<T>,
{
    fn contains(&self, target: T) -> bool { self.0.contains(target) }
}

impl<F, D, N> Function<D> for Ln<N>
where
    F: Scalar + num_traits::real::Real,
    D: Database,
    N: Function<D, Value = F>,
{
    type Error = N::Error;
    type Value = OwnedOf<N::Value>;

    fn evaluate<DR: AsRef<D>>(&self, db: DR) -> Result<Self::Value, Self::Error> {
        self.0.evaluate(db).map(|buffer| buffer.map(|x| x.ln()))
    }
}

impl<T, N> Differentiable<T> for Ln<N>
where
    T: Identifier,
    N: Differentiable<T> + Clone,
{
    type Adjoint = Div<N::Adjoint, N>;

    fn adjoint(&self, target: T) -> Self::Adjoint { Div(self.0.adjoint(target), self.0.clone()) }
}

impl<N: fmt::Display> fmt::Display for Ln<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "ln({})", self.0) }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct SafeXlnX<N>(pub N);

impl<N> Node for SafeXlnX<N> {}

impl<T, N> Contains<T> for SafeXlnX<N>
where
    T: Identifier,
    N: Contains<T>,
{
    fn contains(&self, target: T) -> bool { self.0.contains(target) }
}

impl<F, D, N> Function<D> for SafeXlnX<N>
where
    F: Scalar + num_traits::real::Real,
    D: Database,
    N: Function<D, Value = F>,
{
    type Error = N::Error;
    type Value = OwnedOf<N::Value>;

    fn evaluate<DR: AsRef<D>>(&self, db: DR) -> Result<Self::Value, Self::Error> {
        self.0.evaluate(db).map(|buffer| {
            buffer.map(|x| {
                if x <= num_traits::zero() {
                    num_traits::zero()
                } else {
                    x * x.ln()
                }
            })
        })
    }
}

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

impl<N: fmt::Display> fmt::Display for SafeXlnX<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}) \u{2218} ln({})", self.0, self.0)
    }
}
