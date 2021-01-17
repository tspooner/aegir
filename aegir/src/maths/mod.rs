macro_rules! impl_unary {
    ($name:ident[$str:tt]: $field_type:path, $eval:expr, $grad:expr) => {
        #[derive(Clone, Copy, Debug, Node, Contains)]
        pub struct $name<N>(#[op] pub N);

        impl<D, N> crate::Function<D> for $name<N>
        where
            D: crate::Database,
            N: crate::Function<D>,

            crate::buffer::FieldOf<N::Codomain>: $field_type,
        {
            type Codomain = crate::buffer::OwnedOf<N::Codomain>;
            type Error = N::Error;

            fn evaluate(&self, db: &D) -> Result<Self::Codomain, Self::Error> {
                self.0.evaluate(db).map(|buffer| {
                    crate::buffer::Buffer::map(buffer, $eval)
                })
            }
        }

        impl<D, T, N> crate::Differentiable<D, T> for $name<N>
        where
            D: crate::Database,
            T: crate::Identifier,
            N: crate::Differentiable<D, T>,

            N::Jacobian: crate::buffer::Buffer<Field = crate::buffer::FieldOf<N::Codomain>>,

            crate::buffer::FieldOf<N::Codomain>: $field_type,
        {
            type Jacobian = crate::buffer::OwnedOf<N::Jacobian>;

            fn grad(&self, db: &D, target: T) -> Result<Self::Jacobian, Self::Error> {
                self.0.grad(db, target).map(|buffer| {
                    crate::buffer::Buffer::map(buffer, $grad)
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

macro_rules! impl_trait {
    (@binary $name:ident[$str:tt], $($path:ident)::+, $eval:expr, $grad:expr) => {
        #[derive(Clone, Copy, Debug, Node, Contains)]
        pub struct $name<N1, N2>(#[op] pub N1, #[op] pub N2);

        impl<D, N1, N2> Function<D> for $name<N1, N2>
        where
            D: crate::Database,
            N1: crate::Function<D>,
            N2: crate::Function<D>,

            N1::Codomain: $($path)::+<N2::Codomain>,
            <N1::Codomain as $($path)::+<N2::Codomain>>::Output: Buffer,
        {
            type Codomain = <N1::Codomain as $($path)::+<N2::Codomain>>::Output;
            type Error = either::Either<N1::Error, N2::Error>;

            fn evaluate(&self, db: &D) -> Result<Self::Codomain, Self::Error> {
                self.0.evaluate(db).map_err(either::Either::Left).and_then(|x| {
                    self.1
                        .evaluate(db)
                        .map(|y| $eval(x, y))
                        .map_err(either::Either::Right)
                })
            }
        }

        impl<D, T, N1, N2> Differentiable<D, T> for $name<N1, N2>
        where
            D: crate::Database,
            T: crate::Identifier,
            N1: crate::Differentiable<D, T>,
            N2: crate::Differentiable<D, T>,

            N1::Codomain: $($path)::+<N2::Codomain>,
            <N1::Codomain as $($path)::+<N2::Codomain>>::Output: Buffer,

            N1::Jacobian: $($path)::+<N2::Jacobian>,
            <N1::Jacobian as $($path)::+<N2::Jacobian>>::Output: Buffer<
                Field = crate::buffer::FieldOf<<N1::Codomain as $($path)::+<N2::Codomain>>::Output>
            >,
        {
            type Jacobian = <N1::Jacobian as $($path)::+<N2::Jacobian>>::Output;

            fn grad(&self, db: &D, target: T) -> Result<Self::Jacobian, Self::Error> {
                self.0.grad(db, target).map_err(either::Either::Left).and_then(|x| {
                    self.1
                        .grad(db, target)
                        .map(|y| $grad(x, y))
                        .map_err(either::Either::Right)
                })
            }
        }

        impl<N1: std::fmt::Display, N2: std::fmt::Display> std::fmt::Display for $name<N1, N2> {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "{} {} {}", self.0, $str, self.1)
            }
        }
    }
}

type AddOut<A, B> = <A as std::ops::Add<B>>::Output;
type MulOut<A, B> = <A as std::ops::Mul<B>>::Output;

mod arithmetic;
pub use self::arithmetic::{
    Neg, Dirac, Sign, Abs,
    Add, Sub, Double,
    Mul, Power, Square,
};

mod linalg;
pub use self::linalg::{
    InnerProduct,
    OuterProduct, OuterProductTrait,
    MatMul, MatMulTrait,
};

mod trig;
pub use self::trig::{
    Cos, Cosh, ArcCos, ArcCosh,
    Sin, Sinh, ArcSin, ArcSinh,
    Tan, Tanh, ArcTan, ArcTanh,
};

mod reduce;
pub use self::reduce::Reduce;

mod sigmoid;
pub use self::sigmoid::Sigmoid;

mod special;
pub use self::special::{
    Gamma, LogGamma, Factorial, Erf,
};
