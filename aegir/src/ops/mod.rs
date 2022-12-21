//! Module for concrete operator implementations.
macro_rules! impl_unary {
    ($(#[$attr:meta])* $name:ident[$str:tt]: $field_type:path, $eval:expr, $grad:expr) => {
        $(#[$attr])*
        #[derive(Clone, Copy, Debug, PartialEq, Contains)]
        pub struct $name<N>(#[op] pub N);

        impl<D, N, F> crate::Function<D> for $name<N>
        where
            D: crate::Database,
            N: crate::Function<D>,
            F: crate::buffers::Scalar + $field_type,

            N::Value: crate::buffers::Buffer<Field = F>,
        {
            type Error = N::Error;
            type Value = N::Value;

            fn evaluate<DR: AsRef<D>>(&self, db: DR) -> Result<Self::Value, Self::Error> {
                self.0.evaluate(db).map(|buffer| {
                    crate::buffers::Buffer::map(buffer, $eval)
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

mod arithmetic;
pub use self::arithmetic::{
    Abs,
    Add,
    AddOne,
    Dirac,
    Div,
    Double,
    Mul,
    Negate,
    OneSub,
    Power,
    Sign,
    Square,
    Sub,
    SubOne,
    Sum,
};

mod linalg;
pub use self::linalg::{Contract, TensorDot, TensorProduct};

mod logarithmic;
pub use self::logarithmic::SafeXlnX;

mod trig;
pub use self::trig::{
    ArcCos,
    ArcCosh,
    ArcSin,
    ArcSinh,
    ArcTan,
    ArcTanh,
    Cos,
    Cosh,
    Sin,
    Sinh,
    Tan,
    Tanh,
};

mod sigmoid;
pub use self::sigmoid::Sigmoid;

mod special;
pub use self::special::{Erf, Factorial, Gamma, LogGamma};
