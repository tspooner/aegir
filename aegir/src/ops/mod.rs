macro_rules! impl_unary {
    ($name:ident[$str:tt]: $field_type:path, $eval:expr, $grad:expr) => {
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

            crate::buffer::FieldOf<N::Codomain>: $field_type,
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

            N::Jacobian: crate::buffer::Buffer<Field = crate::buffer::FieldOf<N::Codomain>>,

            crate::buffer::FieldOf<N::Codomain>: $field_type,
        {
            type Jacobian = crate::buffer::OwnedOf<N::Jacobian>;

            fn grad(&self, state: &S, target: T) -> Result<Self::Jacobian, Self::Error> {
                self.0.grad(state, target).map(|buffer| {
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
        new_op!($name<N1, N2>);

        impl<T, N1, N2> crate::Contains<T> for $name<N1, N2>
        where
            T: crate::Identifier,
            N1: crate::Contains<T>,
            N2: crate::Contains<T>,
        {
            fn contains(&self, target: T) -> bool {
                self.0.contains(target) || self.1.contains(target)
            }
        }

        impl<S, N1, N2> Function<S> for $name<N1, N2>
        where
            S: crate::State,
            N1: crate::Function<S>,
            N2: crate::Function<S>,

            N1::Codomain: $($path)::+<N2::Codomain>,
            <N1::Codomain as $($path)::+<N2::Codomain>>::Output: Buffer,
        {
            type Codomain = <N1::Codomain as $($path)::+<N2::Codomain>>::Output;
            type Error = either::Either<N1::Error, N2::Error>;

            fn evaluate(&self, state: &S) -> Result<Self::Codomain, Self::Error> {
                self.0.evaluate(state).map_err(either::Either::Left).and_then(|x| {
                    self.1
                        .evaluate(state)
                        .map(|y| $eval(x, y))
                        .map_err(either::Either::Right)
                })
            }
        }

        impl<S, T, N1, N2> Differentiable<S, T> for $name<N1, N2>
        where
            S: crate::State,
            T: crate::Identifier,
            N1: crate::Differentiable<S, T>,
            N2: crate::Differentiable<S, T>,

            N1::Codomain: $($path)::+<N2::Codomain>,
            <N1::Codomain as $($path)::+<N2::Codomain>>::Output: Buffer,

            N1::Jacobian: $($path)::+<N2::Jacobian>,
            <N1::Jacobian as $($path)::+<N2::Jacobian>>::Output: Buffer<
                Field = crate::buffer::FieldOf<<N1::Codomain as $($path)::+<N2::Codomain>>::Output>
            >,
        {
            type Jacobian = <N1::Jacobian as $($path)::+<N2::Jacobian>>::Output;

            fn grad(&self, state: &S, target: T) -> Result<Self::Jacobian, Self::Error> {
                self.0.grad(state, target).map_err(either::Either::Left).and_then(|x| {
                    self.1
                        .grad(state, target)
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

pub mod arithmetic;
pub mod linalg;
pub mod trig;
pub mod reduce;
pub mod sigmoid;
pub mod special;
