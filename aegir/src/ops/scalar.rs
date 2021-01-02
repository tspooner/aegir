use crate::{
    Function, Differentiable, Node, Identifier,
    buffer::{Buffer, OwnedOf, FieldOf},
    ops::{AddOut, MulOut},
};
use num_traits::{real::Real, Pow};
use std::{fmt, ops};

impl_real!(@unary Neg["-"], |x| { -x }, |dx| { -dx });

impl_trait!(@binary Add["+"], ops::Add, |x, y| { x + y }, |dx, dy| { dx + dy });
impl_trait!(@binary Sub["-"], ops::Sub, |x, y| { x - y }, |dx, dy| { dx - dy });

#[derive(Copy, Clone, Debug)]
pub struct Abs<N>(pub N);

impl<N> Node for Abs<N> {}

impl<N: Function<S>, S> Function<S> for Abs<N>
where
    <N::Codomain as Buffer>::Field: num_traits::real::Real,
{
    type Codomain = OwnedOf<N::Codomain>;
    type Error = N::Error;

    fn evaluate(&self, state: &S) -> Result<Self::Codomain, Self::Error> {
        self.0.evaluate(state).map(|buffer| buffer.map(|x| x.abs()))
    }
}

impl<N, ID: Identifier, S> Differentiable<ID, S> for Abs<N>
where
    N: Differentiable<ID, S, Jacobian = <N as Function<S>>::Codomain>,

    <N::Codomain as Buffer>::Field: num_traits::real::Real,
{
    type Jacobian = OwnedOf<N::Jacobian>;

    fn grad(&self, target: ID, state: &S) -> Result<Self::Jacobian, Self::Error> {
        self.0.dual(target, state).map(|d| {
            d.value.merge(&d.adjoint, |v, &dv| v.signum() * dv)
        })
    }
}

impl<X: fmt::Display> fmt::Display for Abs<X> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "|{}|", self.0)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct Mul<N1, N2>(pub N1, pub N2);

impl<N1, N2> Node for Mul<N1, N2> {}

impl<N1, N2, S> Function<S> for Mul<N1, N2>
where
    N1: Function<S>,
    N2: Function<S>,

    N1::Codomain: ops::Mul<N2::Codomain>,

    MulOut<N1::Codomain, N2::Codomain>: Buffer,
{
    type Codomain = MulOut<N1::Codomain, N2::Codomain>;
    type Error = either::Either<N1::Error, N2::Error>;

    fn evaluate(&self, state: &S) -> Result<Self::Codomain, Self::Error> {
        self.0.evaluate(state).map_err(either::Either::Left).and_then(|x| {
            self.1
                .evaluate(state)
                .map(|y| x * y)
                .map_err(either::Either::Right)
        })
    }
}

impl<N1, N2, T, S> Differentiable<T, S> for Mul<N1, N2>
where
    T: Identifier,
    N1: Differentiable<T, S>,
    N2: Differentiable<T, S>,

    N1::Codomain: ops::Mul<N2::Codomain>,
    N2::Jacobian: ops::Mul<N1::Codomain>,
    N1::Jacobian: ops::Mul<N2::Codomain>,

    for<'a> &'a N1::Codomain:
        ops::Mul<&'a N2::Codomain, Output = MulOut<N1::Codomain, N2::Codomain>>,

    MulOut<N1::Codomain, N2::Codomain>: Buffer,
    MulOut<N1::Jacobian, N2::Codomain>: ops::Add<MulOut<N2::Jacobian, N1::Codomain>>,

    AddOut<MulOut<N1::Jacobian, N2::Codomain>, MulOut<N2::Jacobian, N1::Codomain>>: Buffer<
        Field = FieldOf<MulOut<N1::Codomain, N2::Codomain>>
    >,
{
    type Jacobian = <
        MulOut<N1::Jacobian, N2::Codomain> as ops::Add<MulOut<N2::Jacobian, N1::Codomain>>
    >::Output;

    fn grad(&self, target: T, state: &S) -> Result<Self::Jacobian, Self::Error> {
        let d1 = self.0.dual(target, state).map_err(either::Either::Left)?;
        let d2 = self.1.dual(target, state).map_err(either::Either::Right)?;

        Ok(d1.adjoint * d2.value + d2.adjoint * d1.value)
    }

    fn dual(&self, target: T, state: &S) -> Result<
        crate::dual::Dual<Self::Codomain, Self::Jacobian>,
        Self::Error
    >
    {
        let d1 = self.0.dual(target, state).map_err(either::Either::Left)?;
        let d2 = self.1.dual(target, state).map_err(either::Either::Right)?;

        Ok(crate::dual::Dual {
            value: &d1.value * &d2.value,
            adjoint: d1.adjoint * d2.value + d2.adjoint * d1.value,
        })
    }
}

impl<N1: fmt::Display, N2: Node + fmt::Display> fmt::Display for Mul<N1, N2> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} {}", self.0, self.1)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct Power<N, P>(pub N, pub P);

impl<N, P> Node for Power<N, P> {}

impl<N: Function<S>, P: Clone, S> Function<S> for Power<N, P>
where
    <N::Codomain as Buffer>::Field: num_traits::Pow<P, Output = <N::Codomain as Buffer>::Field>,
{
    type Codomain = OwnedOf<N::Codomain>;
    type Error = N::Error;

    fn evaluate(&self, state: &S) -> Result<Self::Codomain, Self::Error> {
        self.0.evaluate(state).map(|buffer| buffer.map(|x| x.pow(self.1.clone())))
    }
}

impl<N, P, ID, S> Differentiable<ID, S> for Power<N, P>
where
    N: Differentiable<ID, S, Jacobian = <N as Function<S>>::Codomain>,
    P: Clone + num_traits::One + std::ops::Sub<P, Output = P>,
    ID: Identifier,

    <N::Codomain as Buffer>::Field:
        num_traits::Pow<P, Output = <N::Codomain as Buffer>::Field>
        + std::ops::Mul<Output = <N::Codomain as Buffer>::Field>
        + std::ops::Mul<P, Output = <N::Codomain as Buffer>::Field>,
{
    type Jacobian = OwnedOf<N::Jacobian>;

    fn grad(&self, target: ID, state: &S) -> Result<Self::Jacobian, Self::Error> {
        self.0.dual(target, state).map(|d| {
            d.value.merge(&d.adjoint, |v, &g| {
                v.pow(self.1.clone() - num_traits::one()) * self.1.clone() * g
            })
        })
    }
}

impl<X: fmt::Display, P: fmt::Display> fmt::Display for Power<X, P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({})^{}", self.0, self.1)
    }
}
