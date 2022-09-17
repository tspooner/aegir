use crate::{
    buffers::{Buffer, FieldOf, OwnedOf},
    Contains,
    Database,
    Function,
};
use special_fun::FloatSpecial;
use std::fmt;

macro_rules! impl_special {
    (@unary $name:ident, $eval:expr, $grad:expr) => {
        #[derive(Clone, Copy, Debug, PartialEq, Node, Contains)]
        pub struct $name<N>(#[op] pub N);

        impl<D, N> Function<D> for $name<N>
        where
            D: Database,
            N: Function<D>,

            FieldOf<N::Value>: special_fun::FloatSpecial,
        {
            type Value = OwnedOf<N::Value>;
            type Error = N::Error;

            fn evaluate<DR: AsRef<D>>(&self, db: DR) -> Result<Self::Value, Self::Error> {
                self.0.evaluate(db).map(|buffer| buffer.map($eval))
            }
        }
    };
    (@unary $name:ident[$str:tt], $eval:expr, $grad:expr) => {
        impl_special!(@unary $name, $eval, $grad);

        impl<X: fmt::Display + PartialEq> fmt::Display for $name<X> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{}({})", $str, self.0)
            }
        }
    }
}

impl_special!(@unary Gamma["\u{0393}"], |x| x.gamma(), |x| x.gamma() * x.digamma());
impl_special!(@unary LogGamma["ln \u{0393}"], |x| x.loggamma(), |x| x.digamma());

impl_special!(@unary Factorial, |x| x.factorial(), |_| todo!());

impl<X: fmt::Display + PartialEq> fmt::Display for Factorial<X> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "{}!", self.0) }
}

#[derive(Clone, Copy, Debug, PartialEq, Node, Contains)]
pub struct Erf<N>(#[op] pub N);

impl<N> Erf<N> {
    pub fn complementary(self) -> crate::ops::Negate<Self> { crate::ops::Negate(self) }
}

impl<D, N> Function<D> for Erf<N>
where
    D: Database,
    N: Function<D>,

    crate::buffers::FieldOf<N::Value>: special_fun::FloatSpecial,
{
    type Error = N::Error;
    type Value = crate::buffers::OwnedOf<N::Value>;

    fn evaluate<DR: AsRef<D>>(&self, db: DR) -> Result<Self::Value, Self::Error> {
        self.0.evaluate(db).map(|buffer| buffer.map(|x| x.erf()))
    }
}

// impl<D, T, N> Differentiable<D, T> for Erf<N>
// where
// D: Database,
// T: Identifier,
// N: Differentiable<D, T>,

// N::Adjoint: Buffer<Field = crate::buffers::FieldOf<N::Value>>,

// crate::buffers::FieldOf<N::Value>:
// special_fun::FloatSpecial + num_traits::real::Real +
// num_traits::float::FloatConst, {
// type Adjoint = crate::buffers::OwnedOf<N::Adjoint>;

// fn grad(&self, db: &D, target: T) -> Result<Self::Adjoint, Self::Error> {
// self.0.grad(db, target).map(|buffer| {
// buffer.map(|x| {
// let two =
// num_traits::one::<crate::buffers::FieldOf<N::Value>>() + num_traits::one();

// (-x.powi(2)).exp() * two / <crate::buffers::FieldOf<N::Value>>::PI().sqrt()
// })
// })
// }
// }

impl<X: fmt::Display + PartialEq> fmt::Display for Erf<X> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "erf({})", self.0) }
}
