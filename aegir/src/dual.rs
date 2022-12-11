use crate::buffers::{Buffer, Scalar, Class, FieldOf};
use std::ops;

type AddOut<A, B> = <A as std::ops::Add<B>>::Output;
type MulOut<A, B> = <A as std::ops::Mul<B>>::Output;

/// Helper macro to construct a [Dual] variable.
#[macro_export]
macro_rules! dual {
    ($v:expr) => {
        Dual::constant($v)
    };
    ($v:expr, $a:expr) => {
        Dual {
            value: $v,
            adjoint: $a,
        }
    };
}

/// Dual variable for forward-mode autodifferentiation over aegir
/// [buffers](Buffer).
///
/// This struct can be used as a simple container for a value and its adjoint
/// (as we do within this crate). Alternatively, it can be used as a lightweight
/// implementation of the "augmented algebra" commonly found in
/// autodifferentiation libraries. It is by no means exhaustive, but may come in
/// handy for some applications.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Dual<V, A = V> {
    pub value: V,
    pub adjoint: A,
}

impl<F: Scalar, B: Buffer<Field = F>> Dual<B> {
    pub fn variable(buffer: B) -> Dual<B> {
        let adjoint = <B::Class as Class<B::Shape>>::full(buffer.shape(), F::one());

        Dual { value: buffer, adjoint }
    }

    pub fn constant(buffer: B) -> Dual<B> {
        let adjoint = <B::Class as Class<B::Shape>>::full(buffer.shape(), F::zero());

        Dual { value: buffer, adjoint }
    }
}

impl<V, A> From<(V, A)> for Dual<V, A> {
    #[inline]
    fn from((value, adjoint): (V, A)) -> Dual<V, A> { Dual { value, adjoint } }
}

impl<V: Buffer, A: Buffer> Dual<V, A> {
    #[inline]
    fn map<V_, A_>(self, f: impl Fn(V, A) -> (V_, A_)) -> Dual<V_, A_> {
        f(self.value, self.adjoint).into()
    }

    #[inline]
    fn map_ref<V_, A_>(&self, f: impl Fn(&V, &A) -> (V_, A_)) -> Dual<V_, A_> {
        f(&self.value, &self.adjoint).into()
    }
}

impl<V, A: Buffer> Dual<V, A>
where
    FieldOf<A>: std::ops::Neg<Output = FieldOf<A>>,
{
    pub fn conj(self) -> Dual<V, A> {
        Dual {
            value: self.value,
            adjoint: self.adjoint.map(|a| -a),
        }
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Negate
///////////////////////////////////////////////////////////////////////////////////////////////////
impl<V, A> ops::Neg for Dual<V, A>
where
    V: Buffer + std::ops::Neg,
    A: Buffer + std::ops::Neg,
{
    type Output = Dual<V::Output, A::Output>;

    fn neg(self) -> Dual<V::Output, A::Output> { self.map(|v, a| (-v, -a)) }
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Addition
///////////////////////////////////////////////////////////////////////////////////////////////////
impl<V, A> ops::Add<V> for Dual<V, A>
where
    V: Buffer + std::ops::Add<V>,
{
    type Output = Dual<V::Output, A>;

    fn add(self, rhs: V) -> Dual<V::Output, A> {
        Dual {
            value: self.value + rhs,
            adjoint: self.adjoint,
        }
    }
}

impl<'a, V, A> ops::Add<&'a V> for Dual<V, A>
where
    V: Buffer + std::ops::Add<&'a V>,
{
    type Output = Dual<V::Output, A>;

    fn add(self, rhs: &'a V) -> Dual<V::Output, A> {
        Dual {
            value: self.value + rhs,
            adjoint: self.adjoint,
        }
    }
}

impl<V, A> ops::Add<Dual<V, A>> for Dual<V, A>
where
    V: Buffer + std::ops::Add<V>,
    A: Buffer + std::ops::Add<A>,
{
    type Output = Dual<V::Output, A::Output>;

    fn add(self, rhs: Dual<V, A>) -> Dual<V::Output, A::Output> {
        Dual {
            value: self.value + rhs.value,
            adjoint: self.adjoint + rhs.adjoint,
        }
    }
}

impl<'a, V, A> ops::Add<&'a Dual<V, A>> for Dual<V, A>
where
    V: Buffer + std::ops::Add<&'a V>,
    A: Buffer + std::ops::Add<&'a A>,
{
    type Output = Dual<V::Output, A::Output>;

    fn add(self, rhs: &'a Dual<V, A>) -> Dual<V::Output, A::Output> {
        Dual {
            value: self.value + &rhs.value,
            adjoint: self.adjoint + &rhs.adjoint,
        }
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Subtraction
///////////////////////////////////////////////////////////////////////////////////////////////////
impl<V, A> ops::Sub<V> for Dual<V, A>
where
    V: Buffer + std::ops::Sub<V>,
{
    type Output = Dual<V::Output, A>;

    fn sub(self, rhs: V) -> Dual<V::Output, A> {
        Dual {
            value: self.value - rhs,
            adjoint: self.adjoint,
        }
    }
}

impl<'a, V, A> ops::Sub<&'a V> for Dual<V, A>
where
    V: Buffer + std::ops::Sub<&'a V>,
{
    type Output = Dual<V::Output, A>;

    fn sub(self, rhs: &'a V) -> Dual<V::Output, A> {
        Dual {
            value: self.value - rhs,
            adjoint: self.adjoint,
        }
    }
}

impl<V, A> ops::Sub<Dual<V, A>> for Dual<V, A>
where
    V: Buffer + std::ops::Sub<V>,
    A: Buffer + std::ops::Sub<A>,
{
    type Output = Dual<V::Output, A::Output>;

    fn sub(self, rhs: Dual<V, A>) -> Dual<V::Output, A::Output> {
        Dual {
            value: self.value - rhs.value,
            adjoint: self.adjoint - rhs.adjoint,
        }
    }
}

impl<'a, V, A> ops::Sub<&'a Dual<V, A>> for Dual<V, A>
where
    V: Buffer + std::ops::Sub<&'a V>,
    A: Buffer + std::ops::Sub<&'a A>,
{
    type Output = Dual<V::Output, A::Output>;

    fn sub(self, rhs: &'a Dual<V, A>) -> Dual<V::Output, A::Output> {
        Dual {
            value: self.value - &rhs.value,
            adjoint: self.adjoint - &rhs.adjoint,
        }
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Multiplication
///////////////////////////////////////////////////////////////////////////////////////////////////
impl<V, A> ops::Mul<V> for Dual<V, A>
where
    V: Buffer + std::ops::Mul<V>,
{
    type Output = Dual<V::Output, A>;

    fn mul(self, rhs: V) -> Self::Output {
        Dual {
            value: self.value * rhs,
            adjoint: self.adjoint,
        }
    }
}

impl<'a, V, A> ops::Mul<&'a V> for Dual<V, A>
where
    V: Buffer + std::ops::Mul<&'a V>,
{
    type Output = Dual<V::Output, A>;

    fn mul(self, rhs: &'a V) -> Self::Output {
        Dual {
            value: self.value * rhs,
            adjoint: self.adjoint,
        }
    }
}

impl<V, A> ops::Mul<Dual<V, A>> for Dual<V, A>
where
    V: Buffer + std::ops::Mul<V>,
    A: Buffer + std::ops::Mul<V>,

    MulOut<A, V>: std::ops::Add<MulOut<A, V>>,
{
    type Output = Dual<MulOut<V, V>, AddOut<MulOut<A, V>, MulOut<A, V>>>;

    fn mul(self, rhs: Dual<V, A>) -> Self::Output {
        Dual {
            value: self.value.clone() * rhs.value.clone(),
            adjoint: self.adjoint * rhs.value + rhs.adjoint * self.value,
        }
    }
}

impl<'a, V, A> ops::Mul<&'a Dual<V, A>> for Dual<V, A>
where
    V: Buffer + std::ops::Mul<&'a V> + std::ops::Mul<&'a A>,
    A: Buffer + std::ops::Mul<&'a V>,

    MulOut<A, &'a V>: std::ops::Add<MulOut<V, &'a A>>,
{
    type Output = Dual<MulOut<V, &'a V>, AddOut<MulOut<A, &'a V>, MulOut<V, &'a A>>>;

    fn mul(self, rhs: &'a Dual<V, A>) -> Self::Output {
        Dual {
            value: self.value.clone() * &rhs.value,
            adjoint: self.adjoint * &rhs.value + self.value * &rhs.adjoint,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    mod f64 {
        use super::*;

        #[test]
        fn test_neg() {
            assert_eq!(-dual!(0.0), dual!(0.0));
            assert_eq!(-dual!(2.0), dual!(-2.0));
            assert_eq!(-dual!(-4.0), dual!(4.0));
        }

        #[test]
        fn test_add() {
            assert_eq!(dual!(0.0) + dual!(1.0), dual!(1.0));
            assert_eq!(dual!(5.0) + dual!(1.0), dual!(6.0));
            assert_eq!(dual!(5.0) + dual!(-2.0), dual!(3.0));

            assert_eq!(dual!(0.0, 1.0) + dual!(1.0), dual!(1.0, 1.0));
            assert_eq!(dual!(5.0, 1.0) + dual!(1.0), dual!(6.0, 1.0));
            assert_eq!(dual!(5.0, 1.0) + dual!(-2.0), dual!(3.0, 1.0));

            assert_eq!(dual!(0.0) + dual!(1.0, 1.0), dual!(1.0, 1.0));
            assert_eq!(dual!(5.0) + dual!(1.0, 1.0), dual!(6.0, 1.0));
            assert_eq!(dual!(5.0) + dual!(-2.0, 1.0), dual!(3.0, 1.0));

            assert_eq!(dual!(0.0, 1.0) + dual!(1.0, 1.0), dual!(1.0, 2.0));
            assert_eq!(dual!(5.0, 1.0) + dual!(1.0, 1.0), dual!(6.0, 2.0));
            assert_eq!(dual!(5.0, 1.0) + dual!(-2.0, 1.0), dual!(3.0, 2.0));
        }

        #[test]
        fn test_sub() {
            assert_eq!(dual!(0.0) - dual!(1.0), dual!(-1.0));
            assert_eq!(dual!(5.0) - dual!(1.0), dual!(4.0));
            assert_eq!(dual!(5.0) - dual!(-2.0), dual!(7.0));

            assert_eq!(dual!(0.0, 1.0) - dual!(1.0), dual!(-1.0, 1.0));
            assert_eq!(dual!(5.0, 1.0) - dual!(1.0), dual!(4.0, 1.0));
            assert_eq!(dual!(5.0, 1.0) - dual!(-2.0), dual!(7.0, 1.0));

            assert_eq!(dual!(0.0) - dual!(1.0, 1.0), dual!(-1.0, -1.0));
            assert_eq!(dual!(5.0) - dual!(1.0, 1.0), dual!(4.0, -1.0));
            assert_eq!(dual!(5.0) - dual!(-2.0, 1.0), dual!(7.0, -1.0));

            assert_eq!(dual!(0.0, 1.0) - dual!(1.0, 1.0), dual!(-1.0, 0.0));
            assert_eq!(dual!(5.0, 1.0) - dual!(1.0, 1.0), dual!(4.0, 0.0));
            assert_eq!(dual!(5.0, 1.0) - dual!(-2.0, 1.0), dual!(7.0, 0.0));
        }

        #[test]
        fn test_mul() {
            assert_eq!(dual!(0.0) * dual!(1.0), dual!(0.0));
            assert_eq!(dual!(5.0) * dual!(1.0), dual!(5.0));
            assert_eq!(dual!(5.0) * dual!(-2.0), dual!(-10.0));

            assert_eq!(dual!(0.0, 1.0) * dual!(1.0), dual!(0.0, 1.0));
            assert_eq!(dual!(5.0, 1.0) * dual!(1.0), dual!(5.0, 1.0));
            assert_eq!(dual!(5.0, 1.0) * dual!(-2.0), dual!(-10.0, -2.0));

            assert_eq!(dual!(0.0) * dual!(1.0, 1.0), dual!(0.0, 0.0));
            assert_eq!(dual!(5.0) * dual!(1.0, 1.0), dual!(5.0, 5.0));
            assert_eq!(dual!(5.0) * dual!(-2.0, 1.0), dual!(-10.0, 5.0));

            assert_eq!(dual!(0.0, 1.0) - dual!(1.0, 1.0), dual!(-1.0, 0.0));
            assert_eq!(dual!(5.0, 1.0) - dual!(1.0, 1.0), dual!(4.0, 0.0));
            assert_eq!(dual!(5.0, 1.0) - dual!(-2.0, 1.0), dual!(7.0, 0.0));
        }
    }
}
