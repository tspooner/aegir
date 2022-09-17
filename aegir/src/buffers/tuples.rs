use super::{shapes::S1, Buffer, Class, Coalesce, Hadamard, IncompatibleBuffers, OwnedOf, Scalar};

pub struct Tuples;

impl<F: Scalar> Class<S1<2>, F> for Tuples {
    type Buffer = (F, F);

    fn build(_: S1<2>, f: impl Fn(usize) -> F) -> (F, F) { (f(0), f(1)) }

    fn build_subset(
        shape: S1<2>,
        base: F,
        indices: impl Iterator<Item = usize>,
        active: impl Fn(usize) -> F,
    ) -> Self::Buffer {
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

    fn full(_: S1<2>, value: F) -> Self::Buffer { (value, value) }
}

impl<F: Scalar> Buffer for (F, F) {
    type Class = Tuples;
    type Field = F;
    type Shape = S1<2>;

    fn shape(&self) -> Self::Shape { S1 }

    fn to_owned(&self) -> Self { *self }

    fn into_owned(self) -> Self { self }

    fn map<Func: Fn(F) -> Self::Field>(self, f: Func) -> Self { (f(self.0), f(self.1)) }

    fn map_ref<Func: Fn(F) -> Self::Field>(&self, f: Func) -> Self { (f(self.0), f(self.1)) }

    fn fold<Func: Fn(F, &F) -> F>(&self, init: F, f: Func) -> F { f(f(init, &self.0), &self.1) }
}

impl<F: Scalar> Coalesce for (F, F) {
    fn coalesce(
        &self,
        rhs: &(F, F),
        init: F,
        f: impl Fn(F, (F, F)) -> F,
    ) -> Result<F, IncompatibleBuffers<S1<2>>> {
        let x = f(init, (self.0, rhs.0));
        let y = f(x, (self.1, rhs.1));

        Ok(y)
    }
}

impl<F: Scalar> Hadamard for (F, F) {
    type Output = (F, F);

    fn hadamard(
        &self,
        rhs: &(F, F),
        f: impl Fn(F, F) -> F,
    ) -> Result<(F, F), IncompatibleBuffers<S1<2>>> {
        Ok((f(self.0, rhs.0), f(self.1, rhs.1)))
    }
}

impl<F: Scalar> Buffer for &(F, F) {
    type Class = Tuples;
    type Field = F;
    type Shape = S1<2>;

    fn shape(&self) -> Self::Shape { S1 }

    fn map<Func: Fn(F) -> Self::Field>(self, f: Func) -> OwnedOf<Self> { (f(self.0), f(self.1)) }

    fn map_ref<Func: Fn(F) -> Self::Field>(&self, f: Func) -> OwnedOf<Self> {
        (f(self.0), f(self.1))
    }

    fn fold<Func: Fn(F, &F) -> F>(&self, init: F, f: Func) -> F { f(f(init, &self.0), &self.1) }

    fn to_owned(&self) -> OwnedOf<Self> { **self }

    fn into_owned(self) -> OwnedOf<Self> { *self }
}
