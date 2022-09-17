use crate::{
    ops::{Mul, Negate},
    Differentiable,
    Identifier,
};
use num_traits::real::Real;

macro_rules! impl_trig {
    ($name:ident[$str:tt], $eval:expr, $grad:expr) => {
        #[derive(Clone, Copy, Debug, PartialEq, Node, Contains)]
        pub struct $name<N>(#[op] pub N);

        impl<D, N> crate::Function<D> for $name<N>
        where
            D: crate::Database,
            N: crate::Function<D>,

            crate::buffers::FieldOf<N::Value>: num_traits::real::Real,
        {
            type Error = N::Error;
            type Value = crate::buffers::OwnedOf<N::Value>;

            fn evaluate<DR: AsRef<D>>(&self, state: DR) -> Result<Self::Value, Self::Error> {
                self.0
                    .evaluate(state)
                    .map(|buffer| crate::buffers::Buffer::map(buffer, $eval))
            }
        }

        impl<X: std::fmt::Display + PartialEq> std::fmt::Display for $name<X> {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, $str, self.0)
            }
        }
    };
}

impl_trig!(Cos["cos({})"], |x| { x.cos() }, |x| { -x.sin() });

impl<T, N> Differentiable<T> for Cos<N>
where
    T: Identifier,
    N: Differentiable<T> + Clone,
{
    type Adjoint = Negate<Mul<N::Adjoint, Sin<N>>>;

    fn adjoint(&self, ident: T) -> Self::Adjoint {
        Negate(Mul(self.0.adjoint(ident), Sin(self.0.clone())))
    }
}

impl_trig!(Cosh["cosh({})"], |x| { x.cosh() }, |x| { x.sinh() });
impl_trig!(ArcCos["acos({})"], |x| { x.acos() }, |x| {
    (x.powi(2) - num_traits::one()).neg().sqrt().recip().neg()
});
impl_trig!(ArcCosh["acosh({})"], |x| { x.acosh() }, |x| {
    (x.powi(2) - num_traits::one()).sqrt().recip()
});

impl_trig!(Sin["sin({})"], |x| { x.sin() }, |x| { x.cos() });

impl<T, N> Differentiable<T> for Sin<N>
where
    T: Identifier,
    N: Differentiable<T> + Clone,
{
    type Adjoint = Mul<N::Adjoint, Cos<N>>;

    fn adjoint(&self, ident: T) -> Self::Adjoint {
        Mul(self.0.adjoint(ident), Cos(self.0.clone()))
    }
}

impl_trig!(Sinh["sinh({})"], |x| { x.sinh() }, |x| { x.cosh() });
impl_trig!(ArcSin["asin({})"], |x| { x.asin() }, |x| {
    (x.powi(2) - num_traits::one()).neg().sqrt().recip()
});
impl_trig!(ArcSinh["asinh({})"], |x| { x.asinh() }, |x| {
    (x.powi(2) + num_traits::one()).sqrt().recip()
});

impl_trig!(Tan["tan({})"], |x| { x.tan() }, |x| {
    x.cos().powi(2).recip()
});
impl_trig!(Tanh["tanh({})"], |x| { x.tanh() }, |x| {
    x.cosh().powi(2).recip()
});
impl_trig!(ArcTan["atan({})"], |x| { x.atan() }, |x| {
    (x.powi(2) + num_traits::one()).recip()
});
impl_trig!(ArcTanh["atanh({})"], |x| { x.atanh() }, |x| {
    (x.powi(2) - num_traits::one()).neg().recip()
});
