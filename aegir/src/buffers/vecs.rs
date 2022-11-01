use super::{shapes::SDynamic, Buffer, Class, ZipMap, IncompatibleShapes, Scalar, ZipFold};

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
    ) -> Vec<F>
    {
        let mut buf = Self::full(shape, base);

        for [ix] in indices {
            buf[ix] = active([ix]);
        }

        buf
    }

    fn full<F: Scalar>(shape: SDynamic<1>, value: F) -> Vec<F> { vec![value; shape[0]] }
}

impl<F: Scalar> Buffer for Vec<F> {
    type Class = Vecs;
    type Field = F;
    type Shape = SDynamic<1>;

    fn shape(&self) -> Self::Shape { SDynamic([self.len()]) }

    fn get(&self, ix: [usize; 1]) -> Option<F> {
        if ix[0] < self.len() {
            Some(self[ix[0]])
        } else {
            None
        }
    }

    fn map<A: Scalar, M: Fn(F) -> A>(self, f: M) -> Vec<A> { self.into_iter().map(f).collect() }

    fn map_ref<A: Scalar, M: Fn(F) -> A>(&self, f: M) -> Vec<A> { self.iter().map(|x| f(*x)).collect() }

    fn fold<A, M: Fn(A, F) -> A>(&self, init: A, f: M) -> A {
        self.into_iter().copied().fold(init, f)
    }

    fn to_owned(&self) -> Vec<F> { self.clone() }

    fn into_owned(self) -> Vec<F> { self }
}

impl<F: Scalar> ZipFold for Vec<F> {
    fn zip_fold(
        &self,
        rhs: &Vec<F>,
        mut init: F,
        f: impl Fn(F, (F, F)) -> F,
    ) -> Result<F, IncompatibleShapes<SDynamic<1>>> {
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

                Err(IncompatibleShapes(dx, dy))
            },
        }
    }
}

impl<F: Scalar> ZipMap for Vec<F> {
    fn zip_map(
        mut self,
        rhs: &Vec<F>,
        f: impl Fn(F, F) -> F,
    ) -> Result<Vec<F>, IncompatibleShapes<SDynamic<1>>> {
        match (self.len(), rhs.len()) {
            (nx, ny) if nx == ny => {
                for i in 0..nx {
                    self[i] = f(self[i], rhs[i]);
                }

                Ok(self)
            },
            (nx, ny) => {
                let dx = SDynamic([nx]);
                let dy = SDynamic([ny]);

                Err(IncompatibleShapes(dx, dy))
            },
        }
    }

    fn zip_map_ref(
        &self,
        rhs: &Vec<F>,
        f: impl Fn(F, F) -> F,
    ) -> Result<Vec<F>, IncompatibleShapes<SDynamic<1>>> {
        match (self.len(), rhs.len()) {
            (nx, ny) if nx == ny => {
                let mut out = Vec::with_capacity(nx);

                for i in 0..nx {
                    out.insert(i, f(self[i], rhs[i]));
                }

                Ok(out)
            },
            (nx, ny) => {
                let dx = SDynamic([nx]);
                let dy = SDynamic([ny]);

                Err(IncompatibleShapes(dx, dy))
            },
        }
    }

    fn take_left(lhs: Self) -> Vec<F> { lhs }

    fn take_right(rhs: Self) -> Vec<F> { rhs }
}

impl<F: Scalar> Buffer for &Vec<F> {
    type Class = Vecs;
    type Field = F;
    type Shape = SDynamic<1>;

    fn shape(&self) -> Self::Shape { SDynamic([self.len()]) }

    fn get(&self, ix: [usize; 1]) -> Option<F> {
        if ix[0] < self.len() {
            Some(self[ix[0]])
        } else {
            None
        }
    }

    fn to_owned(&self) -> Vec<F> { self.to_vec() }

    fn into_owned(self) -> Vec<F> { self.to_vec() }

    fn map<A: Scalar, M: Fn(F) -> A>(self, f: M) -> Vec<A> { self.into_iter().map(|x| f(*x)).collect() }

    fn map_ref<A: Scalar, M: Fn(F) -> A>(&self, f: M) -> Vec<A> { self.iter().map(|x| f(*x)).collect() }

    fn fold<A, M: Fn(A, F) -> A>(&self, init: A, f: M) -> A {
        self.into_iter().copied().fold(init, f)
    }
}

// impl<F: Scalar> Class<SDynamic<2>, F> for Vecs {
// type Buffer = Vec<Vec<F>>;

// fn build(shape: SDynamic<2>, f: impl Fn([usize; 2]) -> F) -> Vec<Vec<F>> {
// (0..shape[0])
// .map(|r| (0..shape[1]).map(|c| f([r, c])).collect())
// .collect()
// }

// fn build_subset(
// shape: SDynamic<2>,
// base: F,
// indices: impl Iterator<Item = [usize; 2]>,
// active: impl Fn([usize; 2]) -> F,
// ) -> Self::Buffer {
// let mut buf = Self::full(shape, base);

// for ix in indices {
// buf[ix[0]][ix[1]] = active(ix);
// }

// buf
// }

// fn full(shape: SDynamic<2>, value: F) -> Vec<Vec<F>> {
// (0..shape[0]).map(|_| vec![value; shape[1]]).collect()
// }
// }

// impl<F: Scalar> Buffer for Vec<Vec<F>> {
// type Class = Vecs;
// type Field = F;
// type Shape = SDynamic<2>;

// fn get(&self, ix: [usize; 1]) -> Option<F> {
// if ix[0] < self.len() {
// Some(self[ix[0]])
// } else {
// None
// }
// }

// fn shape(&self) -> Self::Shape {
// let l = self.len();

// if l == 0 {
// SDynamic([0, 0])
// } else {
// SDynamic([l, self[0].len()])
// }
// }

// fn map<Func: Fn(F) -> F>(mut self, f: Func) -> Self {
// let shape = self.shape();

// for r in 0..shape[0] {
// for c in 0..shape[1] {
// self[r][c] = f(self[r][c]);
// }
// }

// self
// }

// fn map_ref<Func: Fn(F) -> F>(&self, f: Func) -> Self {
// self.iter().map(|r| r.map_ref(&f)).collect()
// }

// fn fold<Func: Fn(F, &F) -> F>(&self, init: F, f: Func) -> Self::Field {
// self.iter().flatten().fold(init, f)
// }

// fn to_owned(&self) -> Vec<Vec<F>> { self.clone() }

// fn into_owned(self) -> Vec<Vec<F>> { self }
// }

///////////////////////////////////////////////////////////////////////////////////////////////////
// Slices
///////////////////////////////////////////////////////////////////////////////////////////////////
impl<F: Scalar> Buffer for &[F] {
    type Class = Vecs;
    type Field = F;
    type Shape = SDynamic<1>;

    fn shape(&self) -> Self::Shape { SDynamic([self.len()]) }

    fn get(&self, ix: [usize; 1]) -> Option<F> {
        if ix[0] < self.len() {
            Some(self[ix[0]])
        } else {
            None
        }
    }

    fn map<A: Scalar, M: Fn(F) -> A>(self, f: M) -> Vec<A> { self.into_iter().map(|x| f(*x)).collect() }

    fn map_ref<A: Scalar, M: Fn(F) -> A>(&self, f: M) -> Vec<A> { self.iter().map(|x| f(*x)).collect() }

    fn fold<A, M: Fn(A, F) -> A>(&self, init: A, f: M) -> A {
        self.into_iter().copied().fold(init, f)
    }

    fn to_owned(&self) -> Vec<F> { self.into_iter().copied().collect() }

    fn into_owned(self) -> Vec<F> { self.into_iter().copied().collect() }
}
