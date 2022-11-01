use super::{shapes::S0, Buffer, Class, ZipMap, IncompatibleShapes, ZipFold};
use num_traits::Num;

/// Scalar buffer class.
pub struct Scalars;

impl Class<S0> for Scalars {
    type Buffer<F: Scalar> = F;

    fn build<F: Scalar>(_: S0, f: impl Fn(()) -> F) -> F { f(()) }

    fn build_subset<F: Scalar>(
        shape: S0,
        base: F,
        mut indices: impl Iterator<Item = ()>,
        active: impl Fn(()) -> F,
    ) -> F
    {
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
    Copy + Num + Buffer<Class = Scalars, Shape = S0, Field = Self> + ZipMap<Self> + ZipFold<Self>
{
}

macro_rules! impl_scalar {
    ($F:ty) => {
        impl Buffer for $F {
            type Class = Scalars;
            type Field = $F;
            type Shape = S0;

            fn shape(&self) -> Self::Shape { S0 }

            fn get(&self, _: ()) -> Option<$F> { Some(*self) }

            fn map<F: Scalar, M: Fn($F) -> F>(self, f: M) -> F { f(self) }

            fn map_ref<F: Scalar, M: Fn($F) -> F>(&self, f: M) -> F { f(*self) }

            fn fold<F, M: Fn(F, $F) -> F>(&self, init: F, f: M) -> F { f(init, *self) }

            fn to_owned(&self) -> $F { *self }

            fn into_owned(self) -> $F { self }
        }

        impl Buffer for &$F {
            type Class = Scalars;
            type Field = $F;
            type Shape = S0;

            fn shape(&self) -> Self::Shape { S0 }

            fn get(&self, _: ()) -> Option<$F> { Some(**self) }

            fn map<F: Scalar, M: Fn($F) -> F>(self, f: M) -> F { f(*self) }

            fn map_ref<F: Scalar, M: Fn($F) -> F>(&self, f: M) -> F { f(**self) }

            fn fold<F, M: Fn(F, $F) -> F>(&self, init: F, f: M) -> F { f(init, **self) }

            fn to_owned(&self) -> $F { **self }

            fn into_owned(self) -> $F { *self }
        }

        impl ZipFold for $F {
            fn zip_fold(
                &self,
                rhs: &$F,
                init: $F,
                f: impl Fn($F, ($F, $F)) -> $F,
            ) -> Result<$F, IncompatibleShapes<S0>> {
                let out = f(init, (*self, *rhs));

                Ok(out)
            }
        }

        impl ZipMap for $F {
            fn zip_map(
                self,
                rhs: &$F,
                f: impl Fn($F, $F) -> $F,
            ) -> Result<$F, IncompatibleShapes<S0>> {
                Ok(f(self, *rhs))
            }

            fn zip_map_ref(
                &self,
                rhs: &$F,
                f: impl Fn($F, $F) -> $F,
            ) -> Result<$F, IncompatibleShapes<S0>> {
                Ok(f(*self, *rhs))
            }

            fn take_left(lhs: $F) -> $F { lhs }

            fn take_right(rhs: $F) -> $F { rhs }
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
