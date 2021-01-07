use crate::{
    Function, Differentiable, Node, Identifier,
    buffer::Buffer,
};
use num_traits::{float::FloatConst, real::Real};
use special_fun::FloatSpecial;
use std::fmt;

macro_rules! impl_special {
    (@unary $name:ident, $eval:expr, $grad:expr) => {
        new_op!($name<N>);

        impl<S, N: Function<S>> Function<S> for $name<N>
        where
            crate::buffer::FieldOf<N::Codomain>: special_fun::FloatSpecial,
        {
            type Codomain = crate::buffer::OwnedOf<N::Codomain>;
            type Error = N::Error;

            fn evaluate(&self, state: &S) -> Result<Self::Codomain, Self::Error> {
                self.0.evaluate(state).map(|buffer| buffer.map($eval))
            }
        }

        impl<T, S, N> Differentiable<T, S> for $name<N>
        where
            T: Identifier,
            N: Differentiable<T, S>,

            N::Jacobian: Buffer<Field = crate::buffer::FieldOf<N::Codomain>>,

            crate::buffer::FieldOf<N::Codomain>: special_fun::FloatSpecial,
        {
            type Jacobian = crate::buffer::OwnedOf<N::Jacobian>;

            fn grad(&self, target: T, state: &S) -> Result<Self::Jacobian, Self::Error> {
                self.0.grad(target, state).map(|buffer| buffer.map($grad))
            }
        }
    };
    (@unary $name:ident[$str:tt], $eval:expr, $grad:expr) => {
        impl_special!(@unary $name, $eval, $grad);

        impl<X: fmt::Display> fmt::Display for $name<X> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{}({})", $str, self.0)
            }
        }
    }
}

impl_special!(@unary Gamma["\u{0393}"], |x| x.gamma(), |x| x.gamma() * x.digamma());
impl_special!(@unary LogGamma["ln \u{0393}"], |x| x.loggamma(), |x| x.digamma());

impl_special!(@unary Factorial, |x| x.factorial(), |_| todo!());

impl<X: fmt::Display> fmt::Display for Factorial<X> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}!", self.0)
    }
}

new_op!(Erf<N>);

impl<N> Erf<N> {
    pub fn complementary(self) -> crate::ops::scalar::Neg<Self> {
        crate::ops::scalar::Neg(self)
    }
}

impl<S, N: Function<S>> Function<S> for Erf<N>
where
    crate::buffer::FieldOf<N::Codomain>: special_fun::FloatSpecial,
{
    type Codomain = crate::buffer::OwnedOf<N::Codomain>;
    type Error = N::Error;

    fn evaluate(&self, state: &S) -> Result<Self::Codomain, Self::Error> {
        self.0.evaluate(state).map(|buffer| buffer.map(|x| x.erf()))
    }
}

impl<T, S, N> Differentiable<T, S> for Erf<N>
where
    T: Identifier,
    N: Differentiable<T, S>,

    N::Jacobian: Buffer<Field = crate::buffer::FieldOf<N::Codomain>>,

    crate::buffer::FieldOf<N::Codomain>:
        special_fun::FloatSpecial + num_traits::real::Real + num_traits::float::FloatConst,
{
    type Jacobian = crate::buffer::OwnedOf<N::Jacobian>;

    fn grad(&self, target: T, state: &S) -> Result<Self::Jacobian, Self::Error> {
        self.0.grad(target, state).map(|buffer| buffer.map(|x| {
            let two = num_traits::one::<crate::buffer::FieldOf<N::Codomain>>() + num_traits::one();

            (-x.powi(2)).exp() * two / <crate::buffer::FieldOf<N::Codomain>>::PI().sqrt()
        }))
    }
}

impl<X: fmt::Display> fmt::Display for Erf<X> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "erf({})", self.0)
    }
}
