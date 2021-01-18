use crate::buffer::{Buffer, FieldOf, OwnedBuffer, OwnedOf};
use num_traits::{One, Zero};
use std::ops;

type AddOut<A, B> = <A as std::ops::Add<B>>::Output;
type MulOut<A, B> = <A as std::ops::Mul<B>>::Output;

#[derive(Clone, Copy, Debug)]
pub struct Dual<V, A = V> {
    pub value: V,
    pub adjoint: A,
}

impl<B: OwnedBuffer> Dual<B> {
    pub fn variable(value: B) -> Dual<B>
    where
        B::Field: One,
    {
        let adjoint = value.to_ones();
        let value = value.into_owned();

        Dual { value, adjoint }
    }

    pub fn constant(value: B) -> Dual<B>
    where
        B::Field: Zero,
    {
        let adjoint = value.to_zeroes();
        let value = value.into_owned();

        Dual { value, adjoint }
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

    pub fn to_owned(&self) -> Dual<V::Owned, A::Owned> {
        Dual {
            value: self.value.to_owned(),
            adjoint: self.adjoint.to_owned(),
        }
    }

    pub fn into_owned(self) -> Dual<V::Owned, A::Owned> {
        Dual {
            value: self.value.into_owned(),
            adjoint: self.adjoint.into_owned(),
        }
    }
}

impl<V, A: Buffer> Dual<V, A>
where
    FieldOf<A>: std::ops::Neg<Output = FieldOf<A>>,
{
    pub fn conj(self) -> Dual<V, A::Owned> {
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

impl<V, A> ops::Neg for &Dual<V, A>
where
    V: Buffer,
    V::Owned: std::ops::Neg,

    A: Buffer,
    A::Owned: std::ops::Neg,
{
    type Output = Dual<
        <OwnedOf<V> as std::ops::Neg>::Output,
        <OwnedOf<A> as std::ops::Neg>::Output
    >;

    fn neg(self) -> Self::Output { self.map_ref(|v, a| (-v.to_owned(), -a.to_owned())) }
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
    V::Owned: std::ops::Mul<V::Owned>,

    A: Buffer + std::ops::Mul<V>,

    MulOut<A, V>: std::ops::Add<MulOut<A, V>>,
{
    type Output = Dual<
        MulOut<V::Owned, V::Owned>,
        AddOut<MulOut<A, V>, MulOut<A, V>>
    >;

    fn mul(self, rhs: Dual<V, A>) -> Self::Output {
        Dual {
            value: self.value.to_owned() * rhs.value.to_owned(),
            adjoint: self.adjoint * rhs.value + rhs.adjoint * self.value,
        }
    }
}

impl<'a, V, A> ops::Mul<&'a Dual<V, A>> for Dual<V, A>
where
    V: Buffer + std::ops::Mul<&'a V> + std::ops::Mul<&'a A>,
    V::Owned: std::ops::Mul<&'a V>,

    A: Buffer + std::ops::Mul<&'a V>,

    MulOut<A, &'a V>: std::ops::Add<MulOut<V, &'a A>>,
{
    type Output = Dual<
        MulOut<V::Owned, &'a V>,
        AddOut<MulOut<A, &'a V>, MulOut<V, &'a A>>
    >;

    fn mul(self, rhs: &'a Dual<V, A>) -> Self::Output {
        Dual {
            value: self.value.to_owned() * &rhs.value,
            adjoint: self.adjoint * &rhs.value + self.value * &rhs.adjoint,
        }
    }
}
