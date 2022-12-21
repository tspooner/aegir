use crate::{
    buffers::{Buffer, FieldOf},
    ops::{AddOne, Div, Mul},
    Contains,
    Database,
    Differentiable,
    Function,
    Identifier,
    Node,
};
use num_traits::real::Real;
use std::fmt;

#[derive(Copy, Clone, Debug, PartialEq, Contains)]
pub struct Ln<N>(#[op] pub N);

impl<N: Node> Node for Ln<N> {}

impl<D, N> Function<D> for Ln<N>
where
    D: Database,
    N: Function<D>,

    FieldOf<N::Value>: Real,
{
    type Error = N::Error;
    type Value = N::Value;

    fn evaluate<DR: AsRef<D>>(&self, db: DR) -> Result<Self::Value, Self::Error> {
        self.0.evaluate(db).map(|buf| buf.map(|x| x.ln()))
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

#[derive(Copy, Clone, Debug, PartialEq, Contains)]
pub struct SafeXlnX<N>(#[op] pub N);

impl<N: Node> Node for SafeXlnX<N> {}

impl<D, N> Function<D> for SafeXlnX<N>
where
    D: Database,
    N: Function<D>,

    FieldOf<N::Value>: Real,
{
    type Error = N::Error;
    type Value = N::Value;

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
