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
        rhs: Vec<F>,
        f: M,
    ) -> Result<Vec<A>, IncompatibleShapes<SDynamic<1>>> {
        let buf = self
            .into_iter()
            .zip(rhs.into_iter())
            .map(|(x, y)| f(x, y))
            .collect();

        Ok(buf)
    }

    #[inline]
    fn zip_map_dominate<A: Scalar, M: Fn(F) -> A>(
        self,
        lim: SDynamic<1>,
        f: M,
    ) -> Result<Vec<A>, IncompatibleShapes<SDynamic<1>>> {
        Ok(self.into_iter().take(lim[0]).map(f).collect())
    }

    #[inline]
    fn zip_map_dominate_id(self, _: SDynamic<1>) -> Result<Self, IncompatibleShapes<SDynamic<1>>> {
        Ok(self)
    }
}

impl<F: Scalar> ZipMap<F> for Vec<F> {
    type Output<A: Scalar> = Vec<A>;

    #[inline]
    fn zip_map<A: Scalar, M: Fn(F, F) -> A>(
        self,
        rhs: F,
        f: M,
    ) -> Result<Vec<A>, IncompatibleShapes<SDynamic<1>, S0>> {
        let buf = self.into_iter().map(|x| f(x, rhs)).collect();

        Ok(buf)
    }

    #[inline]
    fn zip_map_dominate<A: Scalar, M: Fn(F) -> A>(
        self,
        _: S0,
        f: M,
    ) -> Result<Vec<A>, IncompatibleShapes<SDynamic<1>, S0>> {
        Ok(self.into_iter().map(f).collect())
    }

    #[inline]
    fn zip_map_dominate_id(self, _: S0) -> Result<Self, IncompatibleShapes<SDynamic<1>, S0>> {
        Ok(self)
    }
}

impl<F: Scalar> ZipMap<Vec<F>> for F {
    type Output<A: Scalar> = Vec<A>;

    #[inline]
    fn zip_map<A: Scalar, M: Fn(F, F) -> A>(
        self,
        rhs: Vec<F>,
        f: M,
    ) -> Result<Vec<A>, IncompatibleShapes<S0, SDynamic<1>>> {
        let buf = rhs.into_iter().map(|x| f(self, x)).collect();

        Ok(buf)
    }

    #[inline]
    fn zip_map_dominate<A: Scalar, M: Fn(F) -> A>(
        self,
        lim: SDynamic<1>,
        f: M,
    ) -> Result<Vec<A>, IncompatibleShapes<S0, SDynamic<1>>> {
        Ok(vec![f(self); lim[0]])
    }

    #[inline]
    fn zip_map_dominate_id(
        self,
        lim: SDynamic<1>,
    ) -> Result<Vec<F>, IncompatibleShapes<S0, SDynamic<1>>> {
        Ok(vec![self; lim[0]])
    }
}
