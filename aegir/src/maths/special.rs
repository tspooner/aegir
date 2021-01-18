use crate::{
    buffer::{Buffer, FieldOf, OwnedOf},
    Contains,
    Database,
    Differentiable,
    Function,
    Identifier,
};
use num_traits::{float::FloatConst, real::Real};
use special_fun::FloatSpecial;
use std::fmt;

macro_rules! impl_special {
    (@unary $name:ident, $eval:expr, $grad:expr) => {
        #[derive(Clone, Copy, Debug, Node, Contains)]
        pub struct $name<N>(#[op] pub N);

        impl<D, N> Function<D> for $name<N>
        where
            D: Database,
            N: Function<D>,

            FieldOf<N::Codomain>: special_fun::FloatSpecial,
        {
            type Codomain = OwnedOf<N::Codomain>;
            type Error = N::Error;

            fn evaluate(&self, db: &D) -> Result<Self::Codomain, Self::Error> {
                self.0.evaluate(db).map(|buffer| buffer.map($eval))
            }
        }

        impl<D, T, N> Differentiable<D, T> for $name<N>
        where
            D: Database,
            T: Identifier,
            N: Differentiable<D, T>,

            N::Jacobian: Buffer<Field = FieldOf<N::Codomain>>,

            FieldOf<N::Codomain>: special_fun::FloatSpecial,
        {
            type Jacobian = OwnedOf<N::Jacobian>;

            fn grad(&self, db: &D, target: T) -> Result<Self::Jacobian, Self::Error> {
                self.0.grad(db, target).map(|buffer| buffer.map($grad))
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
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "{}!", self.0) }
}

#[derive(Clone, Copy, Debug, Node, Contains)]
pub struct Erf<N>(#[op] pub N);

impl<N> Erf<N> {
    pub fn complementary(self) -> crate::maths::Negate<Self> { crate::maths::Negate(self) }
}

impl<D, N> Function<D> for Erf<N>
where
    D: Database,
    N: Function<D>,

    crate::buffer::FieldOf<N::Codomain>: special_fun::FloatSpecial,
{
    type Codomain = crate::buffer::OwnedOf<N::Codomain>;
    type Error = N::Error;

    fn evaluate(&self, db: &D) -> Result<Self::Codomain, Self::Error> {
        self.0.evaluate(db).map(|buffer| buffer.map(|x| x.erf()))
    }
}

impl<D, T, N> Differentiable<D, T> for Erf<N>
where
    D: Database,
    T: Identifier,
    N: Differentiable<D, T>,

    N::Jacobian: Buffer<Field = crate::buffer::FieldOf<N::Codomain>>,

    crate::buffer::FieldOf<N::Codomain>:
        special_fun::FloatSpecial + num_traits::real::Real + num_traits::float::FloatConst,
{
    type Jacobian = crate::buffer::OwnedOf<N::Jacobian>;

    fn grad(&self, db: &D, target: T) -> Result<Self::Jacobian, Self::Error> {
        self.0.grad(db, target).map(|buffer| {
            buffer.map(|x| {
                let two =
                    num_traits::one::<crate::buffer::FieldOf<N::Codomain>>() + num_traits::one();

                (-x.powi(2)).exp() * two / <crate::buffer::FieldOf<N::Codomain>>::PI().sqrt()
            })
        })
    }
}

impl<X: fmt::Display> fmt::Display for Erf<X> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "erf({})", self.0) }
}
