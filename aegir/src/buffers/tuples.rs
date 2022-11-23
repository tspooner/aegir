use super::{shapes::S1, Buffer, Class, IncompatibleShapes, OwnedOf, Scalar, ZipFold, ZipMap};

/// Tuple buffer class.
pub struct Tuples;

impl Class<S1<2>> for Tuples {
    type Buffer<F: Scalar> = (F, F);

    fn build<F: Scalar>(_: S1<2>, f: impl Fn(usize) -> F) -> Self::Buffer<F> { (f(0), f(1)) }

    fn build_subset<F: Scalar>(
        shape: S1<2>,
        base: F,
        indices: impl Iterator<Item = usize>,
        active: impl Fn(usize) -> F,
    ) -> Self::Buffer<F> {
        let mut buf = Self::full(shape, base);

        for ix in indices {
            if ix == 0 {
                buf.0 = active(0);
            } else if ix == 1 {
                buf.1 = active(1);
            }
        }

        buf
    }

    fn full<F: Scalar>(_: S1<2>, value: F) -> Self::Buffer<F> { (value, value) }
}

impl<F: Scalar> Buffer for (F, F) {
    type Class = Tuples;
    type Field = F;
    type Shape = S1<2>;

    fn shape(&self) -> Self::Shape { S1 }

    fn get(&self, ix: usize) -> Option<F> {
        match ix {
            0 => Some(self.0),
            1 => Some(self.1),
            _ => None,
        }
    }

    fn get_unchecked(&self, ix: usize) -> F {
        match ix {
            0 => self.0,
            1 => self.1,
            _ => panic!("Invalid index for tuple."),
        }
    }

    fn to_owned(&self) -> Self { *self }

    fn into_owned(self) -> Self { self }

    fn map<A: Scalar, M: Fn(F) -> A>(self, f: M) -> (A, A) { (f(self.0), f(self.1)) }

    fn map_ref<A: Scalar, M: Fn(F) -> A>(&self, f: M) -> (A, A) { (f(self.0), f(self.1)) }

    fn fold<A, M: Fn(A, F) -> A>(&self, init: A, f: M) -> A { f(f(init, self.0), self.1) }
}

impl<F: Scalar> ZipFold for (F, F) {
    fn zip_fold<A: Scalar, M: Fn(A, (F, F)) -> A>(
        &self,
        rhs: &(F, F),
        init: A,
        f: M,
    ) -> Result<A, IncompatibleShapes<S1<2>>> {
        let x = f(init, (self.0, rhs.0));
        let y = f(x, (self.1, rhs.1));

        Ok(y)
    }
}

impl<F: Scalar> ZipMap for (F, F) {
    type Output<A: Scalar> = (A, A);

    fn zip_map<A: Scalar, M: Fn(F, F) -> A>(
        &self,
        rhs: &(F, F),
        f: M,
    ) -> Result<(A, A), IncompatibleShapes<S1<2>>> {
        Ok((f(self.0, rhs.0), f(self.1, rhs.1)))
    }

    fn zip_map_left<A: Scalar, M: Fn(F) -> A>(&self, _: S1<2>, f: M) -> Result<(A, A), IncompatibleShapes<S1<2>>> {
        Ok((f(self.0), f(self.1)))
    }

    fn zip_map_right<A: Scalar, M: Fn(F) -> A>(_: S1<2>, rhs: &Self, f: M) -> Result<(A, A), IncompatibleShapes<S1<2>>> {
        Ok((f(rhs.0), f(rhs.1)))
    }

    fn zip_map_neither<A: Scalar>(_: S1<2>, _: S1<2>, fill_value: A) -> Result<(A, A), IncompatibleShapes<S1<2>>> {
        Ok((fill_value, fill_value))
    }
}

impl<F: Scalar> Buffer for &(F, F) {
    type Class = Tuples;
    type Field = F;
    type Shape = S1<2>;

    fn shape(&self) -> Self::Shape { S1 }

    fn get(&self, ix: usize) -> Option<F> {
        match ix {
            0 => Some(self.0),
            1 => Some(self.1),
            _ => None,
        }
    }

    fn get_unchecked(&self, ix: usize) -> F {
        match ix {
            0 => self.0,
            1 => self.1,
            _ => panic!("Invalid index for tuple."),
        }
    }

    fn map<A: Scalar, M: Fn(F) -> A>(self, f: M) -> (A, A) { (f(self.0), f(self.1)) }

    fn map_ref<A: Scalar, M: Fn(F) -> A>(&self, f: M) -> (A, A) { (f(self.0), f(self.1)) }

    fn fold<A, M: Fn(A, F) -> A>(&self, init: A, f: M) -> A { f(f(init, self.0), self.1) }

    fn to_owned(&self) -> OwnedOf<Self> { **self }

    fn into_owned(self) -> OwnedOf<Self> { *self }
}
