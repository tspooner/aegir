use super::{shapes::S0, Buffer, Class, Coalesce, Hadamard, IncompatibleBuffers};
use num_traits::Num;

pub struct Scalars;

impl<F: Scalar> Class<S0, F> for Scalars {
    type Buffer = F;

    fn build(_: S0, f: impl Fn(()) -> F) -> F { f(()) }

    fn build_subset(
        shape: S0,
        base: F,
        mut indices: impl Iterator<Item = ()>,
        active: impl Fn(()) -> F,
    ) -> Self::Buffer {
        let next = indices.next();

        if next.is_some() {
            Self::full(shape, active(()))
        } else {
            Self::full(shape, base)
        }
    }
}

/// Trait for numeric types implementing basic scalar operations.
pub trait Scalar:
    Copy
    + Num
    + Buffer<Class = Scalars, Shape = S0, Field = Self>
    + Hadamard<Self, Output = Self>
    + Coalesce<Self>
{
}

macro_rules! impl_scalar {
    ($F:ty) => {
        impl Buffer for $F {
            type Class = Scalars;
            type Field = $F;
            type Shape = S0;

            fn shape(&self) -> Self::Shape { S0 }

            fn map<F: Fn($F) -> Self::Field>(self, f: F) -> $F { f(self) }

            fn map_ref<F: Fn($F) -> Self::Field>(&self, f: F) -> Self { f(*self) }

            fn fold<F: Fn($F, &$F) -> $F>(&self, init: $F, f: F) -> $F { f(init, self) }

            fn to_owned(&self) -> $F { *self }

            fn into_owned(self) -> $F { self }
        }

        impl Buffer for &$F {
            type Class = Scalars;
            type Field = $F;
            type Shape = S0;

            fn shape(&self) -> Self::Shape { S0 }

            fn map<F: Fn($F) -> Self::Field>(self, f: F) -> $F { f(*self) }

            fn map_ref<F: Fn($F) -> Self::Field>(&self, f: F) -> $F { f(**self) }

            fn fold<F: Fn($F, &$F) -> $F>(&self, init: $F, f: F) -> $F { f(init, self) }

            fn to_owned(&self) -> $F { **self }

            fn into_owned(self) -> $F { *self }
        }

        impl Coalesce for $F {
            fn coalesce(
                &self,
                rhs: &$F,
                init: $F,
                f: impl Fn($F, ($F, $F)) -> $F,
            ) -> Result<$F, IncompatibleBuffers<S0>> {
                let out = f(init, (*self, *rhs));

                Ok(out)
            }
        }

        impl Hadamard for $F {
            type Output = $F;

            fn hadamard(
                &self,
                rhs: &$F,
                f: impl Fn($F, $F) -> $F,
            ) -> Result<$F, IncompatibleBuffers<S0>> {
                Ok(f(*self, *rhs))
            }
        }

        impl Scalar for $F {}
    };
}

impl_scalar!(f32);
impl_scalar!(f64);

impl_scalar!(u8);
impl_scalar!(u16);
impl_scalar!(u32);
impl_scalar!(u64);
impl_scalar!(usize);

impl_scalar!(i8);
impl_scalar!(i16);
impl_scalar!(i32);
impl_scalar!(i64);
impl_scalar!(isize);

#[cfg(test)]
mod tests {
    use super::*;

    mod f64 {
        use super::*;

        #[test]
        fn test_field() {
            assert_eq!(1.0 + 2.0, 3.0);
            assert_eq!(1.0 - 2.0, -1.0);
            assert_eq!(1.0 * 2.0, 2.0);
            assert_eq!(1.0 / 2.0, 0.5);
        }
    }
}