use crate::{
    buffer::Buffer,
    ops::MulOut,
};
use num_traits::real::Real;
use std::ops::Neg;

macro_rules! impl_trig {
    ($name:ident[$str:tt], $eval:expr, $grad:expr) => {
        new_op!($name<N>);

        impl<T, N> crate::Contains<T> for $name<N>
        where
            T: crate::Identifier,
            N: crate::Contains<T>,
        {
            fn contains(&self, target: T) -> bool { self.0.contains(target) }
        }

        impl<S, N> crate::Function<S> for $name<N>
        where
            S: crate::State,
            N: crate::Function<S>,

            crate::buffer::FieldOf<N::Codomain>: num_traits::real::Real,
        {
            type Codomain = crate::buffer::OwnedOf<N::Codomain>;
            type Error = N::Error;

            fn evaluate(&self, state: &S) -> Result<Self::Codomain, Self::Error> {
                self.0.evaluate(state).map(|buffer| {
                    crate::buffer::Buffer::map(buffer, $eval)
                })
            }
        }

        impl<S, T, N> crate::Differentiable<S, T> for $name<N>
        where
            S: crate::State,
            T: crate::Identifier,
            N: crate::Differentiable<S, T>,

            crate::buffer::FieldOf<N::Codomain>: num_traits::real::Real,

            N::Jacobian: std::ops::Mul<crate::buffer::OwnedOf<N::Codomain>>,

            MulOut<N::Jacobian, crate::buffer::OwnedOf<N::Codomain>>: crate::buffer::Buffer<
                Field = crate::buffer::FieldOf<N::Codomain>
            >,
        {
            type Jacobian = MulOut<N::Jacobian, crate::buffer::OwnedOf<N::Codomain>>;

            fn grad(&self, state: &S, target: T) -> Result<Self::Jacobian, Self::Error> {
                self.0.dual(state, target).map(|d| {
                    d.adjoint * d.value.map($grad)
                })
            }
        }

        impl<X: std::fmt::Display> std::fmt::Display for $name<X> {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, $str, self.0)
            }
        }
    }
}

impl_trig!(Cos["cos({})"], |x| { x.cos() }, |x| { -x.sin() });
impl_trig!(Cosh["cosh({})"], |x| { x.cosh() }, |x| { x.sinh() });
impl_trig!(ArcCos["acos({})"], |x| { x.acos() }, |x| {
    (x.powi(2) - num_traits::one()).neg().sqrt().recip().neg()
});
impl_trig!(ArcCosh["acosh({})"], |x| { x.acosh() }, |x| {
    (x.powi(2) - num_traits::one()).sqrt().recip()
});

impl_trig!(Sin["sin({})"], |x| { x.sin() }, |x| { x.cos() });
impl_trig!(Sinh["sinh({})"], |x| { x.sinh() }, |x| { x.cosh() });
impl_trig!(ArcSin["asin({})"], |x| { x.asin() }, |x| {
    (x.powi(2) - num_traits::one()).neg().sqrt().recip()
});
impl_trig!(ArcSinh["asinh({})"], |x| { x.asinh() }, |x| {
    (x.powi(2) + num_traits::one()).sqrt().recip()
});

impl_trig!(Tan["tan({})"], |x| { x.tan() }, |x| { x.cos().powi(2).recip() });
impl_trig!(Tanh["tanh({})"], |x| { x.tanh() }, |x| { x.cosh().powi(2).recip() });
impl_trig!(ArcTan["atan({})"], |x| { x.atan() }, |x| {
    (x.powi(2) + num_traits::one()).recip()
});
impl_trig!(ArcTanh["atanh({})"], |x| { x.atanh() }, |x| {
    (x.powi(2) - num_traits::one()).neg().recip()
});
