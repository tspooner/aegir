use num_traits::{Float, Pow};
use std::{fmt, marker::PhantomData};

pub mod buffer;

use buffer::*;

#[derive(Clone, Copy, Debug)]
pub struct Dual<F, DF>(pub F, pub DF);

impl<F, DF> Dual<F, DF> {
    #[inline]
    pub fn map<F_, DF_>(self, f: impl Fn(F, DF) -> (F_, DF_)) -> Dual<F_, DF_> {
        f(self.0, self.1).into()
    }

    #[inline]
    pub fn map_f<F_>(self, f: impl Fn(F) -> F_) -> Dual<F_, DF> {
        Dual(f(self.0), self.1)
    }

    #[inline]
    pub fn map_df<DF_>(self, f: impl Fn(DF) -> DF_) -> Dual<F, DF_> {
        Dual(self.0, f(self.1))
    }
}

impl<F, DF: Iterator> Dual<F, DF> {
    pub fn collect<B>(self) -> Dual<F, B>
    where
        B: std::iter::FromIterator<DF::Item>,
    {
        self.map_df(|df| df.collect())
    }
}

impl<F, DF> From<(F, DF)> for Dual<F, DF> {
    #[inline]
    fn from((f, df): (F, DF)) -> Dual<F, DF> { Dual(f, df) }
}

pub trait Function<Domain> {
    type Codomain: BufferMut;

    fn map(&self, args: Domain) -> Self::Codomain;
}

// pub trait LazyFunction<Domain>: Function<Domain> {
    // type Future: Future<Output = Self::Codomain>;
// }

pub trait Differentiable<Domain>: Function<Domain> {
    type Jacobian: BufferMut;

    fn grad(&self, args: Domain) -> Self::Jacobian { self.dual(args).1 }

    fn dual(&self, args: Domain) -> Dual<Self::Codomain, Self::Jacobian>;
}

// pub trait LazyDifferentiable<Domain>: Differentiable<Domain> {
    // type JacobianFuture: Future<Output = Self::Jacobian>;
// }

#[derive(Copy, Clone, Debug)]
pub struct Broadcast<B: Buffer>(pub B::Elem, PhantomData<B>);

impl<B: BufferMut> Function<()> for Broadcast<B> {
    type Codomain = B;

    fn map(&self, _: ()) -> B { B::id(self.0) }
}

impl<B: BufferMut> Differentiable<()> for Broadcast<B> {
    type Jacobian = B;

    fn dual(&self, _: ()) -> Dual<Self::Codomain, Self::Jacobian> {
        Dual(self.map(()), B::id(self.0))
    }
}

impl<B: Buffer> fmt::Display for Broadcast<B>
where
    B::Elem: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct Constant<T>(pub T);

impl<T: Float> Constant<T> {
    pub fn broadcast<B: Buffer<Elem = T>>(self) -> Broadcast<B> {
        Broadcast(self.0, PhantomData)
    }
}

impl<T: BufferMut> Function<()> for Constant<T> {
    type Codomain = T;

    fn map(&self, _: ()) -> T { self.0.clone() }
}

impl<T: BufferMut> Differentiable<()> for Constant<T> {
    type Jacobian = T;

    fn dual(&self, _: ()) -> Dual<Self::Codomain, Self::Jacobian> {
        let df = self.0.map(|_| num_traits::zero());

        Dual(self.0.clone(), df)
    }
}

impl<T: fmt::Display> fmt::Display for Constant<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct Variable;

impl<T: BufferMut> Function<(T,)> for Variable {
    type Codomain = T;

    fn map(&self, (t,): (T,)) -> T { t }
}

impl<T: BufferMut> Differentiable<(T,)> for Variable {
    type Jacobian = T;

    fn dual(&self, (t,): (T,)) -> Dual<Self::Codomain, Self::Jacobian> {
        let df = t.map(|_| num_traits::one());

        Dual(t, df)
    }
}

impl fmt::Display for Variable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "x")
    }
}

#[derive(Copy, Clone, Debug)]
pub struct Neg<X>(pub X);

impl<T, X> Function<T> for Neg<X>
where
    X: Function<T>,
{
    type Codomain = X::Codomain;

    fn map(&self, args: T) -> Self::Codomain {
        self.0.map(args).map_into(|x| -x)
    }
}

impl<T, X> Differentiable<T> for Neg<X>
where
    X: Differentiable<T>,
{
    type Jacobian = X::Jacobian;

    fn dual(&self, args: T) -> Dual<Self::Codomain, Self::Jacobian> {
        self.0.dual(args).map(|f, df| (
            f.map_into(|x| -x),
            df.map_into(|x| -x)
        ))
    }
}

impl<X: fmt::Display> fmt::Display for Neg<X> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "-{}", self.0)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct Cosine<X>(pub X);

impl<T, X> Function<T> for Cosine<X>
where
    X: Function<T>,
{
    type Codomain = X::Codomain;

    fn map(&self, args: T) -> Self::Codomain {
        self.0
            .map(args)
            .map_into(|x| x.cos())
    }
}

impl<T, X> Differentiable<T> for Cosine<X>
where
    X: Differentiable<T>,
{
    type Jacobian = X::Jacobian;

    fn dual(&self, args: T) -> Dual<Self::Codomain, Self::Jacobian> {
        self.0
            .dual(args)
            .map_f(|f| f.map_into(|x| x.cos()))
            .map_df(|df| df.map_into(|x| -x.sin()))
    }
}

impl<X: fmt::Display> fmt::Display for Cosine<X> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "cos({})", self.0)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct Power<X, P>(pub X, pub P);

impl<T, X, P> Function<T> for Power<X, P>
where
    X: Function<T>,
    P: Clone,

    <X::Codomain as Buffer>::Elem: Pow<P, Output = <X::Codomain as Buffer>::Elem>,
{
    type Codomain = X::Codomain;

    fn map(&self, args: T) -> Self::Codomain {
        self.0
            .map(args)
            .map_into(|x| x.pow(self.1.clone()))
    }
}

impl<T, X, P> Differentiable<T> for Power<X, P>
where
    X: Differentiable<T>,
    P: Clone,

    <X::Codomain as Buffer>::Elem: Pow<P, Output = <X::Codomain as Buffer>::Elem>,
    <X::Jacobian as Buffer>::Elem: std::ops::Mul<P, Output = <X::Jacobian as Buffer>::Elem>,
{
    type Jacobian = X::Jacobian;

    fn dual(&self, args: T) -> Dual<Self::Codomain, Self::Jacobian> {
        self.0
            .dual(args)
            .map_f(|f| f.map_into(|x| x.pow(self.1.clone())))
            .map_df(|df| df.map_into(|x| x * self.1.clone()))
    }
}

impl<X: fmt::Display, P: fmt::Display> fmt::Display for Power<X, P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({})^{}", self.0, self.1)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct Sum<X, Y>(pub X, pub Y);

impl<T, U, X, Y> Function<(T, U)> for Sum<X, Y>
where
    X: Function<T>,
    Y: Function<U, Codomain = X::Codomain>,
{
    type Codomain = X::Codomain;

    fn map(&self, (t, u): (T, U)) -> Self::Codomain {
        let out_y = self.1.map(u);

        self.0.map(t).merge_into(&out_y, |x, y| x + y)
    }
}

impl<T, U, X, Y> Differentiable<(T, U)> for Sum<X, Y>
where
    X: Differentiable<T>,
    Y: Differentiable<U, Codomain = X::Codomain>,

    Y::Jacobian: Buffer<Elem = <X::Jacobian as Buffer>::Elem>,
{
    type Jacobian = (X::Jacobian, Y::Jacobian);

    fn dual(&self, (t, u): (T, U)) -> Dual<Self::Codomain, Self::Jacobian> {
        let dual_x = self.0.dual(t);
        let dual_y = self.1.dual(u);

        Dual(
            dual_x.0.merge_into(&dual_y.0, |x, y| x + y),
            (dual_x.1, dual_y.1)
        )
    }
}

impl<T, X, Y> Function<(T,)> for Sum<X, Y>
where
    X: Function<(T,)>,
    Y: Function<(), Codomain = X::Codomain>,
{
    type Codomain = X::Codomain;

    fn map(&self, args: (T,)) -> Self::Codomain {
        let out_y = self.1.map(());

        self.0.map(args).merge_into(&out_y, |x, y| x + y)
    }
}

impl<T, X, Y> Differentiable<(T,)> for Sum<X, Y>
where
    X: Differentiable<(T,)>,
    Y: Function<(), Codomain = X::Codomain>,
{
    type Jacobian = X::Jacobian;

    fn dual(&self, args: (T,)) -> Dual<Self::Codomain, Self::Jacobian> {
        let c = self.1.map(());

        self.0.dual(args).map_f(|f| f.merge_into(&c, |x, y| x + y))
    }
}

impl<X: fmt::Display, Y: fmt::Display> fmt::Display for Sum<X, Y> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} + {}", self.0, self.1)
    }
}
