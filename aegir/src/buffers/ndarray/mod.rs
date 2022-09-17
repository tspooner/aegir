use super::{Buffer, Coalesce, Hadamard, IncompatibleBuffers, Scalar, Shape};

impl<F, S> Buffer for ndarray::ArrayBase<S, Ix1>
where
    F: Scalar,
    S: ndarray::Data<Elem = F>,
{
    type Field = F;
    type Owned = ndarray::Array1<F>;
    type Shape = Shape<1>;

    fn shape(&self) -> Self::Shape { Shape([self.len()]) }

    fn map<Func: Fn(F) -> F>(self, f: Func) -> ndarray::Array1<F> { self.mapv(f) }

    fn map_ref<Func: Fn(F) -> F>(&self, f: Func) -> ndarray::Array1<F> { todo!() }

    fn fold<Func: Fn(F, &F) -> F>(&self, init: F, f: Func) -> F { self.fold(init, f) }

    fn to_owned(&self) -> ndarray::Array1<F> { self.to_owned() }

    fn into_owned(self) -> ndarray::Array1<F> { self.into_owned() }
}

impl<F, S1, S2> Coalesce<ndarray::ArrayBase<S2, Ix1>> for ndarray::ArrayBase<S1, Ix1>
where
    F: Scalar + 'static,
    S1: ndarray::Data<Elem = F>,
    S2: ndarray::Data<Elem = F>,
{
    fn coalesce(
        &self,
        rhs: &ndarray::ArrayBase<S2, Ix1>,
        mut init: F,
        f: impl Fn(F, (F, F)) -> F,
    ) -> Result<F, IncompatibleBuffers<Shape<1>>> {
        match (self.len(), rhs.len()) {
            (nx, ny) if nx == ny => {
                for i in 0..nx {
                    init = f(init, (self[i], rhs[i]));
                }

                Ok(init)
            },
            (nx, ny) => {
                let dx = Shape([nx]);
                let dy = Shape([ny]);

                Err(IncompatibleBuffers(dx, dy))
            },
        }
    }
}

impl<F, S1, S2> Hadamard<ndarray::ArrayBase<S2, Ix1>> for ndarray::ArrayBase<S1, Ix1>
where
    F: Scalar + 'static,
    S1: ndarray::Data<Elem = F>,
    S2: ndarray::Data<Elem = F>,
{
    type Output = ndarray::Array1<F>;

    fn hadamard(
        &self,
        rhs: &ndarray::ArrayBase<S2, Ix1>,
        f: impl Fn(F, F) -> F,
    ) -> Result<Self::Output, IncompatibleBuffers<Shape<1>>> {
        match (self.len(), rhs.len()) {
            (nx, ny) if nx == ny => Ok(ndarray::Array1::from_shape_fn(nx, |i| f(self[i], rhs[i]))),
            (nx, ny) => {
                let dx = Shape([nx]);
                let dy = Shape([ny]);

                Err(IncompatibleBuffers(dx, dy))
            },
        }
    }
}

impl<F, S> Buffer for &ndarray::ArrayBase<S, Ix1>
where
    F: Scalar,
    S: ndarray::Data<Elem = F>,
{
    type Field = F;
    type Owned = ndarray::Array1<F>;
    type Shape = Shape<1>;

    fn shape(&self) -> Self::Shape { Shape([self.len()]) }

    fn map<Func: Fn(F) -> F>(self, f: Func) -> ndarray::Array1<F> { self.into_owned().mapv(f) }

    fn map_ref<Func: Fn(F) -> F>(&self, f: Func) -> ndarray::Array1<F> { todo!() }

    fn fold<Func: Fn(F, &F) -> F>(&self, init: F, f: Func) -> F {
        ndarray::ArrayBase::fold(self, init, f)
    }

    fn to_owned(&self) -> ndarray::Array1<F> { (*self).to_owned() }

    fn into_owned(self) -> ndarray::Array1<F> { ndarray::ArrayBase::to_owned(self) }
}

impl<F, S> Buffer for ndarray::ArrayBase<S, ndarray::Ix2>
where
    F: Scalar,
    S: ndarray::Data<Elem = F> + ndarray::RawDataClone,
{
    type Field = F;
    type Owned = ndarray::Array2<F>;
    type Shape = Shape<2>;

    fn shape(&self) -> Self::Shape {
        let (d1, d2) = self.dim();

        Shape([d1, d2])
    }

    fn map<Func: Fn(F) -> F>(self, f: Func) -> ndarray::Array2<F> { self.into_owned().mapv(f) }

    fn map_ref<Func: Fn(F) -> F>(&self, f: Func) -> ndarray::Array2<F> { todo!() }

    fn fold<Func: Fn(F, &F) -> F>(&self, init: F, f: Func) -> F { self.fold(init, f) }

    fn to_owned(&self) -> ndarray::Array2<F> { self.to_owned() }

    fn into_owned(self) -> ndarray::Array2<F> { self.into_owned() }
}

impl<F, S> Buffer for &ndarray::ArrayBase<S, ndarray::Ix2>
where
    F: Scalar,
    S: ndarray::Data<Elem = F> + ndarray::RawDataClone,
{
    type Field = F;
    type Owned = ndarray::Array2<F>;
    type Shape = Shape<2>;

    fn shape(&self) -> Self::Shape {
        let (d1, d2) = self.dim();

        Shape([d1, d2])
    }

    fn map<Func: Fn(F) -> F>(self, f: Func) -> ndarray::Array2<F> { self.into_owned().mapv(f) }

    fn map_ref<Func: Fn(F) -> F>(&self, f: Func) -> ndarray::Array2<F> { todo!() }

    fn fold<Func: Fn(F, &F) -> F>(&self, init: F, f: Func) -> F {
        ndarray::ArrayBase::fold(self, init, f)
    }

    fn to_owned(&self) -> ndarray::Array2<F> { (*self).to_owned() }

    fn into_owned(self) -> ndarray::Array2<F> { ndarray::ArrayBase::into_owned(self.clone()) }
}
