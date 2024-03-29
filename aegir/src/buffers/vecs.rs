use super::{
    shapes::{SDynamic, Shaped, S0},
    Buffer,
    Class,
    IncompatibleShapes,
    Scalar,
    ZipFold,
    ZipMap,
};

///////////////////////////////////////////////////////////////////////////////////////////////////
// Vecs
///////////////////////////////////////////////////////////////////////////////////////////////////
/// `Vec` buffer class.
pub struct Vecs;

impl Class<SDynamic<1>> for Vecs {
    type Buffer<F: Scalar> = Vec<F>;

    fn build<F: Scalar>(shape: SDynamic<1>, f: impl Fn([usize; 1]) -> F) -> Vec<F> {
        (0..shape[0]).map(|i| f([i])).collect()
    }

    fn build_subset<F: Scalar>(
        shape: SDynamic<1>,
        base: F,
        indices: impl Iterator<Item = [usize; 1]>,
        active: impl Fn([usize; 1]) -> F,
    ) -> Vec<F> {
        let mut buf = Self::full(shape, base);

        for [ix] in indices {
            buf[ix] = active([ix]);
        }

        buf
    }

    fn full<F: Scalar>(shape: SDynamic<1>, value: F) -> Vec<F> { vec![value; shape[0]] }
}

impl<F: Scalar> Shaped for Vec<F> {
    type Shape = SDynamic<1>;

    fn shape(&self) -> Self::Shape { SDynamic([self.len()]) }
}

impl<F: Scalar> Buffer for Vec<F> {
    type Class = Vecs;
    type Field = F;

    fn class() -> Vecs { Vecs }

    fn get_unchecked(&self, ix: [usize; 1]) -> F { self[ix[0]] }

    fn map<A: Scalar, M: Fn(F) -> A>(self, f: M) -> Vec<A> { self.into_iter().map(f).collect() }

    fn map_ref<A: Scalar, M: Fn(F) -> A>(&self, f: M) -> Vec<A> {
        self.iter().map(|x| f(*x)).collect()
    }

    fn mutate<M: Fn(F) -> F>(&mut self, f: M) {
        for i in 0..self.len() {
            self[i] = f(self[i]);
        }
    }

    fn fold<A, M: Fn(A, F) -> A>(&self, init: A, f: M) -> A {
        self.into_iter().copied().fold(init, f)
    }
}

impl<F: Scalar> ZipFold for Vec<F> {
    fn zip_fold<A: Scalar, M: Fn(A, (F, F)) -> A>(
        &self,
        rhs: &Vec<F>,
        mut init: A,
        f: M,
    ) -> Result<A, IncompatibleShapes<SDynamic<1>>> {
        match (self.len(), rhs.len()) {
            (nx, ny) if nx == ny => {
                for i in 0..nx {
                    init = f(init, (self[i], rhs[i]));
                }

                Ok(init)
            },
            (nx, ny) => {
                let dx = SDynamic([nx]);
                let dy = SDynamic([ny]);

                Err(IncompatibleShapes {
                    left: dx,
                    right: dy,
                })
            },
        }
    }
}

impl<F: Scalar> ZipMap for Vec<F> {
    type Output<A: Scalar> = Vec<A>;

    #[inline]
    fn zip_map<A: Scalar, M: Fn(F, F) -> A>(
        self,
        rhs: &Vec<F>,
        f: M,
    ) -> Result<Vec<A>, IncompatibleShapes<SDynamic<1>>> {
        let lshape = self.shape();
        let rshape = rhs.shape();

        if lshape != rshape {
            return Err(IncompatibleShapes {
                left: lshape,
                right: rshape,
            });
        }

        let buf = self
            .into_iter()
            .zip(rhs.into_iter().copied())
            .map(|(x, y)| f(x, y))
            .collect();

        Ok(buf)
    }

    #[inline]
    fn zip_shape(
        self,
        rshape: Self::Shape,
    ) -> Result<Vec<F>, IncompatibleShapes<SDynamic<1>>> {
        let lshape = self.shape();

        if lshape != rshape {
            return Err(IncompatibleShapes {
                left: lshape,
                right: rshape,
            });
        }

        Ok(self)
    }
}

impl<F: Scalar> ZipMap<F> for Vec<F> {
    type Output<A: Scalar> = Vec<A>;

    #[inline]
    fn zip_map<A: Scalar, M: Fn(F, F) -> A>(
        self,
        rhs: &F,
        f: M,
    ) -> Result<Vec<A>, IncompatibleShapes<SDynamic<1>, S0>> {
        let buf = self.into_iter().map(|x| f(x, *rhs)).collect();

        Ok(buf)
    }

    #[inline]
    fn zip_shape(
        self,
        _: S0,
    ) -> Result<Vec<F>, IncompatibleShapes<SDynamic<1>, S0>> {
        Ok(self)
    }
}

impl<F: Scalar> ZipMap<Vec<F>> for F {
    type Output<A: Scalar> = Vec<A>;

    #[inline]
    fn zip_map<A: Scalar, M: Fn(F, F) -> A>(
        self,
        rhs: &Vec<F>,
        f: M,
    ) -> Result<Vec<A>, IncompatibleShapes<S0, SDynamic<1>>> {
        let buf = rhs.into_iter().copied().map(|x| f(self, x)).collect();

        Ok(buf)
    }

    #[inline]
    fn zip_shape(
        self,
        rshape: SDynamic<1>,
    ) -> Result<Vec<F>, IncompatibleShapes<S0, SDynamic<1>>> {
        Ok(Vecs::full(rshape, self))
    }
}
