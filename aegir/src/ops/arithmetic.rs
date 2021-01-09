use crate::{
    Identifier, State, Node, Contains, Function, Differentiable, Compile,
    buffer::{Buffer, Field, OwnedOf, FieldOf},
    sources::Constant,
    ops::{AddOut, MulOut},
};
use num_traits::{real::Real, Pow};
use std::{fmt, ops};

///////////////////////////////////////////////////////////////////////////////////////////////////
// Unary Operators:
///////////////////////////////////////////////////////////////////////////////////////////////////
impl_unary!(Neg["-{}"]: num_traits::real::Real, |x| { -x }, |dx| { -dx });

impl<T, N> Compile<T> for Neg<N>
where
    T: Identifier,
    N: Compile<T>,
{
    type CompiledJacobian = Neg<N::CompiledJacobian>;
    type Error = N::Error;

    fn compile_grad(&self, target: T) -> Result<Self::CompiledJacobian, Self::Error> {
        self.0.compile_grad(target).map(Neg)
    }
}

fn dirac<F: Field + num_traits::Float>(x: F) -> F {
    match x {
        _ if (x == num_traits::zero()) => num_traits::Float::infinity(),
        _ => num_traits::zero(),
    }
}

impl_unary!(Dirac["\u{03B4}({})"]: num_traits::Float, dirac, |_| { num_traits::zero() });

#[derive(Copy, Clone, Debug)]
pub struct Sign<N>(pub N);

impl<N> Node for Sign<N> {}

impl<T, N> Contains<T> for Sign<N>
where
    T: Identifier,
    N: Contains<T>,
{
    fn contains(&self, target: T) -> bool {
        self.0.contains(target)
    }
}

impl<S: State, N: Function<S>> Function<S> for Sign<N>
where
    FieldOf<N::Codomain>: num_traits::real::Real,
{
    type Codomain = OwnedOf<N::Codomain>;
    type Error = N::Error;

    fn evaluate(&self, state: &S) -> Result<Self::Codomain, Self::Error> {
        self.0.evaluate(state).map(|buffer| buffer.map(|x| x.signum()))
    }
}

impl<S, T, N> Differentiable<S, T> for Sign<N>
where
    S: State,
    T: Identifier,
    N: Differentiable<S, T>,

    FieldOf<N::Codomain>: num_traits::Float,
    OwnedOf<N::Codomain>: std::ops::Mul<OwnedOf<N::Jacobian>>,

    MulOut<OwnedOf<N::Codomain>, OwnedOf<N::Jacobian>>: Buffer<
        Field = FieldOf<N::Codomain>,
    >,
{
    type Jacobian = MulOut<OwnedOf<N::Codomain>, OwnedOf<N::Jacobian>>;

    fn grad(&self, state: &S, target: T) -> Result<Self::Jacobian, Self::Error> {
        self.0.dual(state, target).map(|d| {
            d.value.map(dirac) * d.adjoint.map(|g| g)
        })
    }
}

#[derive(Copy, Clone, Debug)]
pub struct Abs<N>(pub N);

impl<N> Node for Abs<N> {}

impl<T, N> Contains<T> for Abs<N>
where
    T: Identifier,
    N: Contains<T>,
{
    fn contains(&self, target: T) -> bool {
        self.0.contains(target)
    }
}

impl<S: State, N: Function<S>> Function<S> for Abs<N>
where
    FieldOf<N::Codomain>: num_traits::real::Real,
{
    type Codomain = OwnedOf<N::Codomain>;
    type Error = N::Error;

    fn evaluate(&self, state: &S) -> Result<Self::Codomain, Self::Error> {
        self.0.evaluate(state).map(|buffer| buffer.map(|x| x.abs()))
    }
}

impl<S, T, N> Differentiable<S, T> for Abs<N>
where
    S: State,
    T: Identifier,
    N: Differentiable<S, T>,

    FieldOf<N::Codomain>: num_traits::real::Real,
    OwnedOf<N::Codomain>: std::ops::Mul<N::Jacobian>,

    MulOut<OwnedOf<N::Codomain>, N::Jacobian>: Buffer<
        Field = FieldOf<N::Codomain>,
    >,
{
    type Jacobian = MulOut<OwnedOf<N::Codomain>, N::Jacobian>;

    fn grad(&self, state: &S, target: T) -> Result<Self::Jacobian, Self::Error> {
        self.0.dual(state, target).map(|d| {
            d.value.map(|v| v.signum()) * d.adjoint
        })
    }
}

impl<X: fmt::Display> fmt::Display for Sign<X> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "sgn({})", self.0)
    }
}

impl<X: fmt::Display> fmt::Display for Abs<X> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "|{}|", self.0)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct Power<N, P>(pub N, pub P);

impl<N, P> Node for Power<N, P> {}

impl<T, N, P> Contains<T> for Power<N, P>
where
    T: Identifier,
    N: Contains<T>,
{
    fn contains(&self, target: T) -> bool {
        self.0.contains(target)
    }
}

impl<S, N, P> Function<S> for Power<N, P>
where
    S: State,
    N: Function<S>,
    P: Field,

    FieldOf<N::Codomain>: num_traits::Pow<P, Output = FieldOf<N::Codomain>>
{
    type Codomain = OwnedOf<N::Codomain>;
    type Error = N::Error;

    fn evaluate(&self, state: &S) -> Result<Self::Codomain, Self::Error> {
        self.0.evaluate(state).map(|buffer| buffer.map(|x| x.pow(self.1.clone())))
    }
}

impl<S, T, N, P> Differentiable<S, T> for Power<N, P>
where
    S: State,
    T: Identifier,
    N: Differentiable<S, T>,
    P: Field + num_traits::One + std::ops::Sub<P, Output = P>,

    FieldOf<N::Codomain>: num_traits::Pow<P, Output = FieldOf<N::Codomain>>,
    FieldOf<N::Jacobian>: std::ops::Mul<P, Output = FieldOf<N::Jacobian>>,
    OwnedOf<N::Codomain>: std::ops::Mul<OwnedOf<N::Jacobian>>,

    MulOut<OwnedOf<N::Codomain>, OwnedOf<N::Jacobian>>: Buffer<
        Field = FieldOf<N::Codomain>,
    >,
{
    type Jacobian = MulOut<OwnedOf<N::Codomain>, OwnedOf<N::Jacobian>>;

    fn grad(&self, state: &S, target: T) -> Result<Self::Jacobian, Self::Error> {
        let np = self.1 - num_traits::one();

        self.0.dual(state, target).map(|d| {
            d.value.map(|v| v.pow(np)) * d.adjoint.map(|g| g * self.1)
        })
    }
}

impl<T, N, P> Compile<T> for Power<N, P>
where
    T: Identifier,
    N: Compile<T> + Clone,
    P: Field,
{
    type CompiledJacobian = Mul<
        Mul<Constant<P>, Power<N, P>>,
        N::CompiledJacobian
    >;
    type Error = N::Error;

    fn compile_grad(&self, target: T) -> Result<Self::CompiledJacobian, Self::Error> {
        let c = Constant(self.1);
        let n = self.0.clone().pow(self.1 - num_traits::one());
        let g = self.0.compile_grad(target)?;

        Ok(c.mul(n).mul(g))
    }
}

impl<X: fmt::Display, P: fmt::Display> fmt::Display for Power<X, P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({})^{}", self.0, self.1)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct Twice<N>(pub N);

impl<N> Node for Twice<N> {}

impl<T, N> Contains<T> for Twice<N>
where
    T: Identifier,
    N: Contains<T>,
{
    fn contains(&self, target: T) -> bool {
        self.0.contains(target)
    }
}

impl<S, N> Function<S> for Twice<N>
where
    S: State,
    N: Function<S>,

    FieldOf<N::Codomain>: std::ops::Mul<
        FieldOf<N::Codomain>,
        Output = FieldOf<N::Codomain>
    >,
{
    type Codomain = OwnedOf<N::Codomain>;
    type Error = N::Error;

    fn evaluate(&self, state: &S) -> Result<Self::Codomain, Self::Error> {
        let two = num_traits::one::<FieldOf<N::Codomain>>() + num_traits::one();

        self.0.evaluate(state).map(|buffer| buffer.map(|x| two * x))
    }
}

impl<S, T, N> Differentiable<S, T> for Twice<N>
where
    S: State,
    T: Identifier,
    N: Differentiable<S, T>,

    FieldOf<N::Jacobian>: std::ops::Mul<
        FieldOf<N::Jacobian>,
        Output = FieldOf<N::Jacobian>
    >,
{
    type Jacobian = OwnedOf<N::Jacobian>;

    fn grad(&self, state: &S, target: T) -> Result<Self::Jacobian, Self::Error> {
        let two = num_traits::one::<FieldOf<N::Jacobian>>() + num_traits::one();

        self.0.dual(state, target).map(|d| d.adjoint.map(|g| two * g))
    }
}

impl<T, N> Compile<T> for Twice<N>
where
    T: Identifier,
    N: Compile<T>,
{
    type CompiledJacobian = Twice<N::CompiledJacobian>;
    type Error = N::Error;

    fn compile_grad(&self, target: T) -> Result<Self::CompiledJacobian, Self::Error> {
        self.0.compile_grad(target).map(Twice)
    }
}

impl<X: fmt::Display> fmt::Display for Twice<X> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "2({})", self.0)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct Squared<N>(pub N);

impl<N> Node for Squared<N> {}

impl<T, N> Contains<T> for Squared<N>
where
    T: Identifier,
    N: Contains<T>,
{
    fn contains(&self, target: T) -> bool {
        self.0.contains(target)
    }
}

impl<S, N> Function<S> for Squared<N>
where
    S: State,
    N: Function<S>,

    FieldOf<N::Codomain>: num_traits::Pow<
        FieldOf<N::Codomain>,
        Output = FieldOf<N::Codomain>
    >,
{
    type Codomain = OwnedOf<N::Codomain>;
    type Error = N::Error;

    fn evaluate(&self, state: &S) -> Result<Self::Codomain, Self::Error> {
        let two = num_traits::one::<FieldOf<N::Codomain>>() + num_traits::one();

        self.0.evaluate(state).map(|buffer| buffer.map(|x| x.pow(two)))
    }
}

impl<S, T, N> Differentiable<S, T> for Squared<N>
where
    S: State,
    T: Identifier,
    N: Differentiable<S, T>,

    FieldOf<N::Codomain>: num_traits::Pow<
        FieldOf<N::Codomain>,
        Output = FieldOf<N::Codomain>
    > + std::ops::Mul<
        FieldOf<N::Codomain>,
        Output = FieldOf<N::Codomain>
    >,

    N::Codomain: std::ops::Mul<OwnedOf<N::Jacobian>>,

    MulOut<N::Codomain, OwnedOf<N::Jacobian>>: Buffer<Field = FieldOf<N::Codomain>>,
{
    type Jacobian = MulOut<N::Codomain, OwnedOf<N::Jacobian>>;

    fn grad(&self, state: &S, target: T) -> Result<Self::Jacobian, Self::Error> {
        let two = num_traits::one::<FieldOf<N::Jacobian>>() + num_traits::one();

        self.0.dual(state, target).map(|d| {
            d.value * d.adjoint.map(|g| two * g)
        })
    }
}

impl<T, N> Compile<T> for Squared<N>
where
    T: Identifier,
    N: Compile<T> + Clone,
{
    type CompiledJacobian = Mul<Twice<N>, N::CompiledJacobian>;
    type Error = N::Error;

    fn compile_grad(&self, target: T) -> Result<Self::CompiledJacobian, Self::Error> {
        let g = self.0.compile_grad(target)?;

        Ok(Twice(self.0.clone()).mul(g))
    }
}

impl<X: fmt::Display> fmt::Display for Squared<X> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({})^2", self.0)
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Binary Operators:
///////////////////////////////////////////////////////////////////////////////////////////////////
impl_trait!(@binary Add["+"], ops::Add, |x, y| { x + y }, |dx, dy| { dx + dy });

impl<T, N1, N2> Compile<T> for Add<N1, N2>
where
    T: Identifier,
    N1: Compile<T>,
    N2: Compile<T>,
{
    type CompiledJacobian = Add<N1::CompiledJacobian, N2::CompiledJacobian>;
    type Error = either::Either<N1::Error, N2::Error>;

    fn compile_grad(&self, target: T) -> Result<Self::CompiledJacobian, Self::Error> {
        let gl = self.0.compile_grad(target).map_err(|e| either::Left(e))?;
        let gr = self.1.compile_grad(target).map_err(|e| either::Right(e))?;

        Ok(gl.add(gr))
    }
}

impl_trait!(@binary Sub["-"], ops::Sub, |x, y| { x - y }, |dx, dy| { dx - dy });

impl<T, N1, N2> Compile<T> for Sub<N1, N2>
where
    T: Identifier,
    N1: Compile<T>,
    N2: Compile<T>,
{
    type CompiledJacobian = Sub<N1::CompiledJacobian, N2::CompiledJacobian>;
    type Error = either::Either<N1::Error, N2::Error>;

    fn compile_grad(&self, target: T) -> Result<Self::CompiledJacobian, Self::Error> {
        let gl = self.0.compile_grad(target).map_err(|e| either::Left(e))?;
        let gr = self.1.compile_grad(target).map_err(|e| either::Right(e))?;

        Ok(gl.sub(gr))
    }
}

#[derive(Copy, Clone, Debug)]
pub struct Mul<N1, N2>(pub N1, pub N2);

impl<N1, N2> Node for Mul<N1, N2> {}

impl<T, N1, N2> Contains<T> for Mul<N1, N2>
where
    T: Identifier,
    N1: Contains<T>,
    N2: Contains<T>,
{
    fn contains(&self, target: T) -> bool {
        self.0.contains(target) || self.1.contains(target)
    }
}

impl<S, N1, N2> Function<S> for Mul<N1, N2>
where
    S: State,
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

impl<S, T, N1, N2> Differentiable<S, T> for Mul<N1, N2>
where
    S: State,
    T: Identifier,
    N1: Differentiable<S, T>,
    N2: Differentiable<S, T>,

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

    fn grad(&self, state: &S, target: T) -> Result<Self::Jacobian, Self::Error> {
        let d1 = self.0.dual(state, target).map_err(either::Either::Left)?;
        let d2 = self.1.dual(state, target).map_err(either::Either::Right)?;

        Ok(d1.adjoint * d2.value + d2.adjoint * d1.value)
    }

    fn dual(&self, state: &S, target: T) -> Result<
        crate::dual::Dual<Self::Codomain, Self::Jacobian>,
        Self::Error
    >
    {
        let d1 = self.0.dual(state, target).map_err(either::Either::Left)?;
        let d2 = self.1.dual(state, target).map_err(either::Either::Right)?;

        Ok(crate::dual::Dual {
            value: &d1.value * &d2.value,
            adjoint: d1.adjoint * d2.value + d2.adjoint * d1.value,
        })
    }
}

impl<T, N1, N2> Compile<T> for Mul<N1, N2>
where
    T: Identifier,
    N1: Compile<T> + Clone,
    N2: Compile<T> + Clone,
{
    type CompiledJacobian = Add<
        Mul<N1::CompiledJacobian, N2>,
        Mul<N2::CompiledJacobian, N1>
    >;
    type Error = either::Either<N1::Error, N2::Error>;

    fn compile_grad(&self, target: T) -> Result<Self::CompiledJacobian, Self::Error> {
        let gl = self.0.compile_grad(target).map_err(|e| either::Left(e))?;
        let gr = self.1.compile_grad(target).map_err(|e| either::Right(e))?;

        let ll = gl.mul(self.1.clone());
        let rr = gr.mul(self.0.clone());

        Ok(ll.add(rr))
    }
}

impl<N1: fmt::Display, N2: Node + fmt::Display> fmt::Display for Mul<N1, N2> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} {}", self.0, self.1)
    }
}
