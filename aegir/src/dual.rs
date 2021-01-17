use crate::buffer::{Buffer, OwnedBuffer, OwnedOf, FieldOf};
use num_traits::{Zero, One};
use std::ops;

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

        Dual { value, adjoint, }
    }

    pub fn constant(value: B) -> Dual<B>
    where
        B::Field: Zero,
    {
        let adjoint = value.to_zeroes();
        let value = value.into_owned();

        Dual { value, adjoint, }
    }
}

impl<B: Buffer> Dual<B> {
    #[inline]
    fn map<B_>(self, f: impl Fn(B, B) -> (B_, B_)) -> Dual<B_> {
        f(self.value, self.adjoint).into()
    }

    #[inline]
    fn map_ref<B_>(&self, f: impl Fn(&B, &B) -> (B_, B_)) -> Dual<B_> {
        f(&self.value, &self.adjoint).into()
    }

    pub fn to_owned(&self) -> Dual<B::Owned> {
        Dual {
            value: self.value.to_owned(),
            adjoint: self.adjoint.to_owned(),
        }
    }

    pub fn into_owned(self) -> Dual<B::Owned> {
        Dual {
            value: self.value.into_owned(),
            adjoint: self.adjoint.into_owned(),
        }
    }
}

impl<B: Buffer> Dual<B>
where
    FieldOf<B>: std::ops::Neg<Output = FieldOf<B>>,
{
    pub fn conj(self) -> Dual<B, B::Owned> {
        Dual {
            value: self.value,
            adjoint: self.adjoint.map(|a| -a),
        }
    }
}

impl<B> ops::Neg for Dual<B>
where
    B: Buffer + std::ops::Neg,
{
    type Output = Dual<B::Output>;

    fn neg(self) -> Dual<B::Output> {
        self.map(|v, a| (-v, -a))
    }
}

impl<B> ops::Neg for &Dual<B>
where
    B: Buffer,
    B::Owned: std::ops::Neg,
{
    type Output = Dual<<OwnedOf<B> as std::ops::Neg>::Output>;

    fn neg(self) -> Self::Output {
        self.map_ref(|v, a| (-v.to_owned(), -a.to_owned()))
    }
}

impl<B> ops::Add<B> for Dual<B>
where
    B: Buffer + std::ops::Add<B>,
{
    type Output = Dual<B::Output, B>;

    fn add(self, rhs: B) -> Dual<B::Output, B> {
        Dual {
            value: self.value + rhs,
            adjoint: self.adjoint,
        }
    }
}

impl<'a, B> ops::Add<&'a B> for Dual<B>
where
    B: Buffer + std::ops::Add<&'a B>,
{
    type Output = Dual<B::Output, B>;

    fn add(self, rhs: &'a B) -> Dual<B::Output, B> {
        Dual {
            value: self.value + rhs,
            adjoint: self.adjoint,
        }
    }
}

impl<B> ops::Add<Dual<B>> for Dual<B>
where
    B: Buffer + std::ops::Add<B>,
{
    type Output = Dual<B::Output>;

    fn add(self, rhs: Dual<B>) -> Dual<B::Output> {
        Dual {
            value: self.value + rhs.value,
            adjoint: self.adjoint + rhs.adjoint,
        }
    }
}

impl<'a, B> ops::Add<&'a Dual<B>> for Dual<B>
where
    B: Buffer + std::ops::Add<&'a B>,
{
    type Output = Dual<B::Output>;

    fn add(self, rhs: &'a Dual<B>) -> Dual<B::Output> {
        Dual {
            value: self.value + &rhs.value,
            adjoint: self.adjoint + &rhs.adjoint,
        }
    }
}

impl<B> ops::Sub<B> for Dual<B>
where
    B: Buffer + std::ops::Sub<B>,
{
    type Output = Dual<B::Output, B>;

    fn sub(self, rhs: B) -> Dual<B::Output, B> {
        Dual {
            value: self.value - rhs,
            adjoint: self.adjoint,
        }
    }
}

impl<'a, B> ops::Sub<&'a B> for Dual<B>
where
    B: Buffer + std::ops::Sub<&'a B>,
{
    type Output = Dual<B::Output, B>;

    fn sub(self, rhs: &'a B) -> Dual<B::Output, B> {
        Dual {
            value: self.value - rhs,
            adjoint: self.adjoint,
        }
    }
}

impl<B> ops::Sub<Dual<B>> for Dual<B>
where
    B: Buffer + std::ops::Sub<B>,
{
    type Output = Dual<B::Output>;

    fn sub(self, rhs: Dual<B>) -> Dual<B::Output> {
        Dual {
            value: self.value - rhs.value,
            adjoint: self.adjoint - rhs.adjoint,
        }
    }
}

impl<'a, B> ops::Sub<&'a Dual<B>> for Dual<B>
where
    B: Buffer + std::ops::Sub<&'a B>,
{
    type Output = Dual<B::Output>;

    fn sub(self, rhs: &'a Dual<B>) -> Dual<B::Output> {
        Dual {
            value: self.value - &rhs.value,
            adjoint: self.adjoint - &rhs.adjoint,
        }
    }
}

impl<B> From<(B, B)> for Dual<B> {
    #[inline]
    fn from((value, adjoint): (B, B)) -> Dual<B> {
        Dual { value, adjoint, }
    }
}
