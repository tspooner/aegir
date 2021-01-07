macro_rules! new_op {
    ($name:ident<$($tp:ident),+>) => {
        #[derive(Copy, Clone, Debug)]
        pub struct $name<$($tp),+>($(pub $tp),+);

        impl<$($tp),+> Node for $name<$($tp),+> {}
    };
    ($name:ident<$($tp:ident),+>($inner:ty)) => {
        #[derive(Copy, Clone, Debug)]
        pub struct $name<$($tp),+>(pub $inner);

        impl<$($tp),+> Node for $name<$($tp),+> {}
    };
}

macro_rules! impl_real {
    (@unary $name:ident[$str:tt], $eval:expr, $grad:expr) => {
        new_op!($name<N>);

        impl<S, N: Function<S>> Function<S> for $name<N>
        where
            crate::buffer::FieldOf<N::Codomain>: num_traits::real::Real,
        {
            type Codomain = crate::buffer::OwnedOf<N::Codomain>;
            type Error = N::Error;

            fn evaluate(&self, state: &S) -> Result<Self::Codomain, Self::Error> {
                self.0.evaluate(state).map(|buffer| buffer.map($eval))
            }
        }

        // impl<T, N: Contains<T>> Function<T> for $name<N> {
            // fn contains(&self) -> bool {
                // self.0.evaluate(state).map(|buffer| buffer.map($eval))
            // }
        // }

        impl<T, S, N> Differentiable<T, S> for $name<N>
        where
            T: Identifier,
            N: Differentiable<T, S>,

            N::Jacobian: Buffer<Field = crate::buffer::FieldOf<N::Codomain>>,

            crate::buffer::FieldOf<N::Codomain>: num_traits::real::Real,
        {
            type Jacobian = crate::buffer::OwnedOf<N::Jacobian>;

            fn grad(&self, target: T, state: &S) -> Result<Self::Jacobian, Self::Error> {
                self.0.grad(target, state).map(|buffer| buffer.map($grad))
            }
        }

        impl<X: fmt::Display> fmt::Display for $name<X> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{}({})", $str, self.0)
            }
        }
    }
}

macro_rules! impl_trait {
    (@binary $name:ident[$str:tt], $($path:ident)::+, $eval:expr, $grad:expr) => {
        new_op!($name<N1, N2>);

        impl<N1, N2, S> Function<S> for $name<N1, N2>
        where
            N1: Function<S>,
            N2: Function<S>,

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

        impl<N1, N2, T, S> Differentiable<T, S> for $name<N1, N2>
        where
            T: Identifier,
            N1: Differentiable<T, S>,
            N2: Differentiable<T, S>,

            N1::Codomain: $($path)::+<N2::Codomain>,
            <N1::Codomain as $($path)::+<N2::Codomain>>::Output: Buffer,

            N1::Jacobian: $($path)::+<N2::Jacobian>,
            <N1::Jacobian as $($path)::+<N2::Jacobian>>::Output: Buffer<
                Field = crate::buffer::FieldOf<<N1::Codomain as $($path)::+<N2::Codomain>>::Output>
            >,
        {
            type Jacobian = <N1::Jacobian as $($path)::+<N2::Jacobian>>::Output;

            fn grad(&self, target: T, state: &S) -> Result<Self::Jacobian, Self::Error> {
                self.0.grad(target, state).map_err(either::Either::Left).and_then(|x| {
                    self.1
                        .grad(target, state)
                        .map(|y| $grad(x, y))
                        .map_err(either::Either::Right)
                })
            }
        }

        impl<N1: fmt::Display, N2: fmt::Display> fmt::Display for $name<N1, N2> {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{} {} {}", self.0, $str, self.1)
            }
        }
    }
}

macro_rules! impl_newtype {
    ($name:ident<$($tp:ident),+>($inner:ty)) => {
        new_op!($name<$($tp),+>($inner));

        impl<$($tp),+, S> crate::Function<S> for InnerProduct<$($tp),+>
        where
            $inner: crate::Function<S>,
        {
            type Codomain = crate::CodomainOf<$inner, S>;
            type Error = crate::ErrorOf<$inner, S>;

            fn evaluate(&self, state: &S) -> Result<Self::Codomain, Self::Error> {
                self.0.evaluate(state)
            }
        }

        impl<$($tp),+, T, S> crate::Differentiable<T, S> for InnerProduct<N1, N2>
        where
            T: crate::Identifier,
            $inner: crate::Differentiable<T, S>,
        {
            type Jacobian = crate::JacobianOf<$inner, T, S>;

            fn grad(&self, target: T, state: &S) -> Result<Self::Jacobian, Self::Error> {
                self.0.grad(target, state)
            }
        }
    }
}

type AddOut<A, B> = <A as std::ops::Add<B>>::Output;
type MulOut<A, B> = <A as std::ops::Mul<B>>::Output;

pub mod scalar;
pub mod linalg;
pub mod trig;
pub mod reduce;
pub mod sigmoid;
pub mod special;
