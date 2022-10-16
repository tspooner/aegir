use super::{
    shapes::{S0, S1, S2, S3, S4, S5},
    Buffer,
    Class,
    ZipMap,
    IncompatibleShapes,
    Scalar,
    ZipFold,
};

/// Array buffer class.
pub struct Arrays;

// S1 ---------------------------------------------------------------------- S1
// S1 ---------------------------------------------------------------------- S1
// S1 ---------------------------------------------------------------------- S1
impl<F: Scalar, const D1: usize> Class<S1<D1>, F> for Arrays {
    type Buffer = [F; D1];

    fn build(_: S1<D1>, f: impl Fn(usize) -> F) -> [F; D1] { array_init::array_init(f) }

    fn build_subset(
        shape: S1<D1>,
        base: F,
        indices: impl Iterator<Item = usize>,
        active: impl Fn(usize) -> F,
    ) -> Self::Buffer {
        let mut buf = Self::full(shape, base);

        for ix in indices {
            buf[ix] = active(ix);
        }

        buf
    }

    fn full(_: S1<D1>, value: F) -> Self::Buffer { [value; D1] }
}

impl<F: Scalar, const D1: usize> Buffer for [F; D1] {
    type Class = Arrays;
    type Field = F;
    type Shape = S1<D1>;

    fn shape(&self) -> Self::Shape { S1 }

    fn get(&self, ix: usize) -> Option<F> {
        if ix < D1 {
            Some(self[ix])
        } else {
            None
        }
    }

    fn map<M: Fn(F) -> F>(self, f: M) -> Self { <[F; D1]>::map(self, |x| x.map(&f)) }

    fn map_ref<M: Fn(F) -> F>(&self, f: M) -> Self { array_init::array_init(|i| f(self[i])) }

    fn fold<A, M: Fn(A, &F) -> A>(&self, mut init: A, f: M) -> A {
        for i in 0..D1 {
            init = self[i].fold(init, &f)
        }

        init
    }

    fn to_owned(&self) -> [F; D1] { self.clone() }

    fn into_owned(self) -> [F; D1] { self }
}

impl<F: Scalar, const D1: usize> ZipMap for [F; D1] {
    fn zip_map(
        mut self,
        rhs: &Self,
        f: impl Fn(F, F) -> F,
    ) -> Result<Self, IncompatibleShapes<Self::Shape>> {
        for i in 0..D1 {
            self[i] = unsafe {
                self[i].zip_map(&rhs[i], &f).unwrap_unchecked()
            };
        }

        Ok(self)
    }

    fn zip_map_ref(
        &self,
        rhs: &Self,
        f: impl Fn(F, F) -> F,
    ) -> Result<Self, IncompatibleShapes<Self::Shape>> {
        Ok(array_init::array_init(|i| unsafe { self[i].zip_map_ref(&rhs[i], &f).unwrap_unchecked() }))
    }

    fn take_left(lhs: Self) -> Self { lhs }

    fn take_right(rhs: Self) -> Self { rhs }
}

// impl<F: Scalar, const D1: usize> ZipMap<F> for [F; D1] {
    // type Output = Self;

    // fn zip_map(
        // mut self,
        // rhs: &F,
        // f: impl Fn(F, F) -> F,
    // ) -> Result<Self, IncompatibleShapes<Self::Shape, S0>> {
        // for i in 0..D1 {
            // self[i] = f(self[i], *rhs);
        // }

        // Ok(self)
    // }

    // fn zip_map_ref(
        // &self,
        // rhs: &F,
        // f: impl Fn(F, F) -> F,
    // ) -> Result<Self, IncompatibleShapes<Self::Shape, S0>> {
        // Ok(array_init::array_init(|i| f(self[i], *rhs)))
    // }

    // fn take_left(lhs: Self, _: F::Shape) -> Result<Self, IncompatibleShapes<Self::Shape, S0>> { Ok(lhs) }

    // fn take_right(_: Self::Shape, rhs: F) -> Result<Self, IncompatibleShapes<Self::Shape, S0>> { Ok([rhs; D1]) }
// }

impl<F: Scalar, const D1: usize> ZipFold for [F; D1] {
    fn zip_fold(
        &self,
        rhs: &Self,
        mut acc: F,
        f: impl Fn(F, (F, F)) -> F,
    ) -> Result<F, IncompatibleShapes<Self::Shape>> {
        for i in 0..D1 {
            acc = f(acc, (self[i], rhs[i]))
        }

        Ok(acc)
    }
}

impl<F: Scalar, const D1: usize> ZipFold<F> for [F; D1] {
    fn zip_fold(
        &self,
        rhs: &F,
        mut acc: F,
        f: impl Fn(F, (F, F)) -> F,
    ) -> Result<F, IncompatibleShapes<Self::Shape, S0>> {
        for i in 0..D1 {
            acc = f(acc, (self[i], *rhs))
        }

        Ok(acc)
    }
}

impl<F: Scalar, const D1: usize> Buffer for &[F; D1] {
    type Class = Arrays;
    type Field = F;
    type Shape = S1<D1>;

    fn shape(&self) -> Self::Shape { S1 }

    fn get(&self, ix: usize) -> Option<F> {
        if ix < D1 {
            Some(self[ix])
        } else {
            None
        }
    }

    fn map<M: Fn(F) -> F>(self, f: M) -> [F; D1] { <[F; D1]>::map_ref(self, |x| x.map(&f)) }

    fn map_ref<M: Fn(F) -> F>(&self, f: M) -> [F; D1] { array_init::array_init(|i| f(self[i])) }

    fn fold<A, M: Fn(A, &F) -> A>(&self, mut init: A, f: M) -> A {
        for i in 0..D1 {
            init = self[i].fold(init, &f)
        }

        init
    }

    fn to_owned(&self) -> [F; D1] { <[F; D1] as Buffer>::to_owned(self) }

    fn into_owned(self) -> [F; D1] { self.clone() }
}

impl<F: Scalar, const D1: usize> ZipMap for &[F; D1] {
    fn zip_map_ref(
        &self,
        rhs: &Self,
        f: impl Fn(F, F) -> F,
    ) -> Result<[F; D1], IncompatibleShapes<Self::Shape>> {
        Ok(array_init::array_init(|i| unsafe { self[i].zip_map(&rhs[i], &f).unwrap_unchecked() }))
    }

    fn take_left(lhs: Self) -> [F; D1] { lhs.into_owned() }

    fn take_right(rhs: Self) -> [F; D1] { rhs.into_owned() }
}

// impl<F: Scalar, const D1: usize> ZipMap<F> for &[F; D1] {
    // type Output = [F; D1];

    // fn zip_map_ref(
        // &self,
        // rhs: &F,
        // f: impl Fn(F, F) -> F,
    // ) -> Result<Self::Output, IncompatibleShapes<Self::Shape, S0>> {
        // Ok(array_init::array_init(|i| unsafe { self[i].zip_map(rhs, &f).unwrap_unchecked() }))
    // }

    // fn take_left(lhs: Self, _: F::Shape) -> Result<Self::Output, IncompatibleShapes<Self::Shape, S0>> { Ok(lhs.into_owned()) }

    // fn take_right(_: Self::Shape, rhs: F) -> Result<Self::Output, IncompatibleShapes<Self::Shape, S0>> { Ok([rhs; D1]) }
// }

impl<F: Scalar, const D1: usize> ZipFold for &[F; D1] {
    fn zip_fold(
        &self,
        rhs: &Self,
        mut acc: F,
        f: impl Fn(F, (F, F)) -> F,
    ) -> Result<F, IncompatibleShapes<Self::Shape>> {
        for i in 0..D1 {
            acc = f(acc, (self[i], rhs[i]))
        }

        Ok(acc)
    }
}

impl<F: Scalar, const D1: usize> ZipFold<F> for &[F; D1] {
    fn zip_fold(
        &self,
        rhs: &F,
        mut acc: F,
        f: impl Fn(F, (F, F)) -> F,
    ) -> Result<F, IncompatibleShapes<Self::Shape, S0>> {
        for i in 0..D1 {
            acc = f(acc, (self[i], *rhs))
        }

        Ok(acc)
    }
}

// S2 ---------------------------------------------------------------------- S2
// S2 ---------------------------------------------------------------------- S2
// S2 ---------------------------------------------------------------------- S2
impl<F: Scalar, const D1: usize, const D2: usize> Class<S2<D1, D2>, F> for Arrays {
    type Buffer = [[F; D2]; D1];

    fn build(_: S2<D1, D2>, f: impl Fn([usize; 2]) -> F) -> [[F; D2]; D1] {
        array_init::array_init(|i| array_init::array_init(|j| f([i, j])))
    }

    fn build_subset(
        shape: S2<D1, D2>,
        base: F,
        indices: impl Iterator<Item = [usize; 2]>,
        active: impl Fn([usize; 2]) -> F,
    ) -> Self::Buffer {
        let mut buf = Self::full(shape, base);

        for [i1, i2] in indices {
            buf[i1][i2] = active([i1, i2]);
        }

        buf
    }

    fn full(_: S2<D1, D2>, value: F) -> Self::Buffer { [[value; D2]; D1] }
}

impl<F: Scalar, const D1: usize, const D2: usize> Buffer for [[F; D2]; D1] {
    type Class = Arrays;
    type Field = F;
    type Shape = S2<D1, D2>;

    fn shape(&self) -> Self::Shape { S2 }

    fn get(&self, ix: [usize; 2]) -> Option<F> {
        if ix[0] < D1 && ix[1] < D2 {
            Some(self[ix[0]][ix[1]])
        } else {
            None
        }
    }

    fn map<M: Fn(F) -> F>(self, f: M) -> Self { Self::map(self, |x| x.map(&f)) }

    fn map_ref<M: Fn(F) -> F>(&self, f: M) -> Self {
        array_init::array_init(|i| self[i].map_ref(&f))
    }

    fn fold<A, M: Fn(A, &F) -> A>(&self, mut init: A, f: M) -> A {
        for i in 0..D1 {
            init = self[i].fold(init, &f)
        }

        init
    }

    fn to_owned(&self) -> Self { self.clone() }

    fn into_owned(self) -> Self { self }
}

impl<F: Scalar, const D1: usize, const D2: usize> ZipMap for [[F; D2]; D1] {
    fn zip_map(
        mut self,
        rhs: &Self,
        f: impl Fn(F, F) -> F,
    ) -> Result<Self, IncompatibleShapes<Self::Shape>> {
        for i in 0..D1 {
            self[i] = unsafe {
                self[i].zip_map(&rhs[i], &f).unwrap_unchecked()
            };
        }

        Ok(self)
    }

    fn zip_map_ref(
        &self,
        rhs: &Self,
        f: impl Fn(F, F) -> F,
    ) -> Result<Self, IncompatibleShapes<Self::Shape>> {
        Ok(array_init::array_init(|i| unsafe { self[i].zip_map_ref(&rhs[i], &f).unwrap_unchecked() }))
    }

    fn take_left(lhs: Self) -> Self { lhs }

    fn take_right(rhs: Self) -> Self { rhs }
}

// impl<F: Scalar, const D1: usize, const D2: usize> ZipMap<F> for [[F; D2]; D1] {
    // type Output = Self;

    // fn zip_map(
        // mut self,
        // rhs: &F,
        // f: impl Fn(F, F) -> F,
    // ) -> Result<Self, IncompatibleShapes<Self::Shape, S0>> {
        // for i in 0..D1 {
            // self[i] = unsafe {
                // self[i].zip_map(rhs, &f).unwrap_unchecked()
            // };
        // }

        // Ok(self)
    // }

    // fn zip_map_ref(
        // &self,
        // rhs: &F,
        // f: impl Fn(F, F) -> F,
    // ) -> Result<Self, IncompatibleShapes<Self::Shape, S0>> {
        // Ok(array_init::array_init(|i| unsafe { self[i].zip_map_ref(rhs, &f).unwrap_unchecked() }))
    // }

    // fn take_left(lhs: Self, _: F::Shape) -> Result<Self, IncompatibleShapes<Self::Shape, S0>> { Ok(lhs) }

    // fn take_right(_: Self::Shape, rhs: F) -> Result<Self, IncompatibleShapes<Self::Shape, S0>> { Ok([[rhs; D2]; D1]) }
// }

impl<F: Scalar, const D1: usize, const D2: usize> ZipFold for [[F; D2]; D1] {
    fn zip_fold(
        &self,
        rhs: &Self,
        mut acc: F,
        f: impl Fn(F, (F, F)) -> F,
    ) -> Result<F, IncompatibleShapes<Self::Shape>> {
        for i in 0..D1 {
            for j in 0..D2 {
                acc = f(acc, (self[i][j], rhs[i][j]))
            }
        }

        Ok(acc)
    }
}

impl<F: Scalar, const D1: usize, const D2: usize> ZipFold<F> for [[F; D2]; D1] {
    fn zip_fold(
        &self,
        rhs: &F,
        mut acc: F,
        f: impl Fn(F, (F, F)) -> F,
    ) -> Result<F, IncompatibleShapes<Self::Shape, S0>> {
        for i in 0..D1 {
            for j in 0..D2 {
                acc = f(acc, (self[i][j], *rhs))
            }
        }

        Ok(acc)
    }
}

impl<F: Scalar, const D1: usize, const D2: usize> Buffer for &[[F; D2]; D1] {
    type Class = Arrays;
    type Field = F;
    type Shape = S2<D1, D2>;

    fn shape(&self) -> Self::Shape { S2 }

    fn get(&self, ix: [usize; 2]) -> Option<F> {
        if ix[0] < D1 && ix[1] < D2 {
            Some(self[ix[0]][ix[1]])
        } else {
            None
        }
    }

    fn map<M: Fn(F) -> F>(self, f: M) -> [[F; D2]; D1] {
        <[[F; D2]; D1]>::map_ref(self, |x| x.map(&f))
    }

    fn map_ref<M: Fn(F) -> F>(&self, f: M) -> [[F; D2]; D1] {
        array_init::array_init(|i| self[i].map_ref(&f))
    }

    fn fold<A, M: Fn(A, &F) -> A>(&self, mut init: A, f: M) -> A {
        for i in 0..D1 {
            init = self[i].fold(init, &f)
        }

        init
    }

    fn to_owned(&self) -> [[F; D2]; D1] { <[[F; D2]; D1] as Buffer>::to_owned(self) }

    fn into_owned(self) -> [[F; D2]; D1] { self.clone() }
}

impl<F: Scalar, const D1: usize, const D2: usize> ZipMap for &[[F; D2]; D1] {
    fn zip_map_ref(
        &self,
        rhs: &Self,
        f: impl Fn(F, F) -> F,
    ) -> Result<[[F; D2]; D1], IncompatibleShapes<Self::Shape>> {
        Ok(array_init::array_init(|i| unsafe { self[i].zip_map(&rhs[i], &f).unwrap_unchecked() }))
    }

    fn take_left(lhs: Self) -> [[F; D2]; D1] { lhs.into_owned() }

    fn take_right(rhs: Self) -> [[F; D2]; D1] { rhs.into_owned() }
}

// impl<F: Scalar, const D1: usize, const D2: usize> ZipMap<F> for &[[F; D2]; D1] {
    // type Output = [[F; D2]; D1];

    // fn zip_map_ref(
        // &self,
        // rhs: &F,
        // f: impl Fn(F, F) -> F,
    // ) -> Result<Self::Output, IncompatibleShapes<Self::Shape, S0>> {
        // Ok(array_init::array_init(|i| unsafe { self[i].zip_map(rhs, &f).unwrap_unchecked() }))
    // }

    // fn take_left(lhs: Self, _: F::Shape) -> Result<Self::Output, IncompatibleShapes<Self::Shape, S0>> { Ok(lhs.into_owned()) }

    // fn take_right(_: Self::Shape, rhs: F) -> Result<Self::Output, IncompatibleShapes<Self::Shape, S0>> { Ok([[rhs; D2]; D1]) }
// }

impl<F: Scalar, const D1: usize, const D2: usize> ZipFold for &[[F; D2]; D1] {
    fn zip_fold(
        &self,
        rhs: &Self,
        mut acc: F,
        f: impl Fn(F, (F, F)) -> F,
    ) -> Result<F, IncompatibleShapes<Self::Shape>> {
        for i in 0..D1 {
            for j in 0..D2 {
                acc = f(acc, (self[i][j], rhs[i][j]))
            }
        }

        Ok(acc)
    }
}

impl<F: Scalar, const D1: usize, const D2: usize> ZipFold<F> for &[[F; D2]; D1] {
    fn zip_fold(
        &self,
        rhs: &F,
        mut acc: F,
        f: impl Fn(F, (F, F)) -> F,
    ) -> Result<F, IncompatibleShapes<Self::Shape, S0>> {
        for i in 0..D1 {
            for j in 0..D2 {
                acc = f(acc, (self[i][j], *rhs))
            }
        }

        Ok(acc)
    }
}

// S3 ---------------------------------------------------------------------- S3
// S3 ---------------------------------------------------------------------- S3
// S3 ---------------------------------------------------------------------- S3
impl<F, const D1: usize, const D2: usize, const D3: usize> Class<S3<D1, D2, D3>, F> for Arrays
where
    F: Scalar,
{
    type Buffer = [[[F; D3]; D2]; D1];

    fn build(_: S3<D1, D2, D3>, f: impl Fn([usize; 3]) -> F) -> [[[F; D3]; D2]; D1] {
        array_init::array_init(|i| {
            array_init::array_init(|j| array_init::array_init(|k| f([i, j, k])))
        })
    }

    fn build_subset(
        shape: S3<D1, D2, D3>,
        base: F,
        indices: impl Iterator<Item = [usize; 3]>,
        active: impl Fn([usize; 3]) -> F,
    ) -> Self::Buffer {
        let mut buf = Self::full(shape, base);

        for [i1, i2, i3] in indices {
            buf[i1][i2][i3] = active([i1, i2, i3]);
        }

        buf
    }

    fn full(_: S3<D1, D2, D3>, value: F) -> Self::Buffer { [[[value; D3]; D2]; D1] }
}

impl<F, const D1: usize, const D2: usize, const D3: usize> Buffer for [[[F; D3]; D2]; D1]
where
    F: Scalar,
{
    type Class = Arrays;
    type Field = F;
    type Shape = S3<D1, D2, D3>;

    fn shape(&self) -> Self::Shape { S3 }

    fn get(&self, ix: [usize; 3]) -> Option<F> {
        if ix[0] < D1 && ix[1] < D2 && ix[2] < D3 {
            Some(self[ix[0]][ix[1]][ix[2]])
        } else {
            None
        }
    }

    fn map<M: Fn(F) -> F>(self, f: M) -> Self { Self::map(self, |x| Buffer::map(x, &f)) }

    fn map_ref<M: Fn(F) -> F>(&self, f: M) -> Self {
        array_init::array_init(|i| self[i].map_ref(&f))
    }

    fn fold<A, M: Fn(A, &F) -> A>(&self, mut init: A, f: M) -> A {
        for i in 0..D1 {
            init = self[i].fold(init, &f)
        }

        init
    }

    fn to_owned(&self) -> Self { self.clone() }

    fn into_owned(self) -> Self { self }
}

impl<F: Scalar, const D1: usize, const D2: usize, const D3: usize> ZipMap
    for [[[F; D3]; D2]; D1]
{
    fn zip_map(
        mut self,
        rhs: &Self,
        f: impl Fn(F, F) -> F,
    ) -> Result<Self, IncompatibleShapes<Self::Shape>> {
        for i in 0..D1 {
            self[i] = unsafe {
                self[i].zip_map(&rhs[i], &f).unwrap_unchecked()
            };
        }

        Ok(self)
    }

    fn zip_map_ref(
        &self,
        rhs: &Self,
        f: impl Fn(F, F) -> F,
    ) -> Result<Self, IncompatibleShapes<Self::Shape>> {
        Ok(array_init::array_init(|i| unsafe { self[i].zip_map_ref(&rhs[i], &f).unwrap_unchecked() }))
    }

    fn take_left(lhs: Self) -> Self { lhs }

    fn take_right(rhs: Self) -> Self { rhs }
}

// impl<F: Scalar, const D1: usize, const D2: usize, const D3: usize> ZipMap<F>
    // for [[[F; D3]; D2]; D1]
// {
    // type Output = Self;

    // fn zip_map(
        // mut self,
        // rhs: &F,
        // f: impl Fn(F, F) -> F,
    // ) -> Result<Self, IncompatibleShapes<Self::Shape, S0>> {
        // for i in 0..D1 {
            // self[i] = unsafe {
                // self[i].zip_map(rhs, &f).unwrap_unchecked()
            // };
        // }

        // Ok(self)
    // }

    // fn zip_map_ref(
        // &self,
        // rhs: &F,
        // f: impl Fn(F, F) -> F,
    // ) -> Result<Self, IncompatibleShapes<Self::Shape, S0>> {
        // Ok(array_init::array_init(|i| unsafe { self[i].zip_map_ref(rhs, &f).unwrap_unchecked() }))
    // }

    // fn take_left(lhs: Self, _: F::Shape) -> Result<Self, IncompatibleShapes<Self::Shape, S0>> { Ok(lhs) }

    // fn take_right(_: Self::Shape, rhs: F) -> Result<Self, IncompatibleShapes<Self::Shape, S0>> { Ok([[[rhs; D3]; D2]; D1]) }
// }

impl<F: Scalar, const D1: usize, const D2: usize, const D3: usize> ZipFold
    for [[[F; D3]; D2]; D1]
{
    fn zip_fold(
        &self,
        rhs: &Self,
        mut acc: F,
        f: impl Fn(F, (F, F)) -> F,
    ) -> Result<F, IncompatibleShapes<Self::Shape>> {
        for i in 0..D1 {
            for j in 0..D2 {
                for k in 0..D3 {
                    acc = f(acc, (self[i][j][k], rhs[i][j][k]))
                }
            }
        }

        Ok(acc)
    }
}

impl<F: Scalar, const D1: usize, const D2: usize, const D3: usize> ZipFold<F>
    for [[[F; D3]; D2]; D1]
{
    fn zip_fold(
        &self,
        rhs: &F,
        mut acc: F,
        f: impl Fn(F, (F, F)) -> F,
    ) -> Result<F, IncompatibleShapes<Self::Shape, S0>> {
        for i in 0..D1 {
            for j in 0..D2 {
                for k in 0..D3 {
                    acc = f(acc, (self[i][j][k], *rhs))
                }
            }
        }

        Ok(acc)
    }
}

impl<F, const D1: usize, const D2: usize, const D3: usize> Buffer for &[[[F; D3]; D2]; D1]
where
    F: Scalar,
{
    type Class = Arrays;
    type Field = F;
    type Shape = S3<D1, D2, D3>;

    fn shape(&self) -> Self::Shape { S3 }

    fn get(&self, ix: [usize; 3]) -> Option<F> {
        if ix[0] < D1 && ix[1] < D2 && ix[2] < D3 {
            Some(self[ix[0]][ix[1]][ix[2]])
        } else {
            None
        }
    }

    fn map<M: Fn(F) -> F>(self, f: M) -> [[[F; D3]; D2]; D1] {
        <[[[F; D3]; D2]; D1]>::map_ref(self, f)
    }

    fn map_ref<M: Fn(F) -> F>(&self, f: M) -> [[[F; D3]; D2]; D1] {
        array_init::array_init(|i| self[i].map_ref(&f))
    }

    fn fold<A, M: Fn(A, &F) -> A>(&self, mut init: A, f: M) -> A {
        for i in 0..D1 {
            init = self[i].fold(init, &f)
        }

        init
    }

    fn to_owned(&self) -> [[[F; D3]; D2]; D1] { <[[[F; D3]; D2]; D1] as Buffer>::to_owned(self) }

    fn into_owned(self) -> [[[F; D3]; D2]; D1] { self.clone() }
}

impl<F: Scalar, const D1: usize, const D2: usize, const D3: usize> ZipMap
    for &[[[F; D3]; D2]; D1]
{
    fn zip_map_ref(
        &self,
        rhs: &Self,
        f: impl Fn(F, F) -> F,
    ) -> Result<[[[F; D3]; D2]; D1], IncompatibleShapes<Self::Shape>> {
        Ok(array_init::array_init(|i| unsafe { self[i].zip_map(&rhs[i], &f).unwrap_unchecked() }))
    }

    fn take_left(lhs: Self) -> [[[F; D3]; D2]; D1] { lhs.into_owned() }

    fn take_right(rhs: Self) -> [[[F; D3]; D2]; D1] { rhs.into_owned() }
}

// impl<F: Scalar, const D1: usize, const D2: usize, const D3: usize> ZipMap<F>
    // for &[[[F; D3]; D2]; D1]
// {
    // type Output = [[[F; D3]; D2]; D1];

    // fn zip_map_ref(
        // &self,
        // rhs: &F,
        // f: impl Fn(F, F) -> F,
    // ) -> Result<Self::Output, IncompatibleShapes<Self::Shape, S0>> {
        // Ok(array_init::array_init(|i| unsafe { self[i].zip_map(rhs, &f).unwrap_unchecked() }))
    // }

    // fn take_left(lhs: Self, _: F::Shape) -> Result<Self::Output, IncompatibleShapes<Self::Shape, S0>> { Ok(lhs.into_owned()) }

    // fn take_right(_: Self::Shape, rhs: F) -> Result<Self::Output, IncompatibleShapes<Self::Shape, S0>> { Ok([[[rhs; D3]; D2]; D1]) }
// }

impl<F: Scalar, const D1: usize, const D2: usize, const D3: usize> ZipFold
    for &[[[F; D3]; D2]; D1]
{
    fn zip_fold(
        &self,
        rhs: &Self,
        mut acc: F,
        f: impl Fn(F, (F, F)) -> F,
    ) -> Result<F, IncompatibleShapes<Self::Shape>> {
        for i in 0..D1 {
            for j in 0..D2 {
                for k in 0..D3 {
                    acc = f(acc, (self[i][j][k], rhs[i][j][k]))
                }
            }
        }

        Ok(acc)
    }
}

impl<F: Scalar, const D1: usize, const D2: usize, const D3: usize> ZipFold<F>
    for &[[[F; D3]; D2]; D1]
{
    fn zip_fold(
        &self,
        rhs: &F,
        mut acc: F,
        f: impl Fn(F, (F, F)) -> F,
    ) -> Result<F, IncompatibleShapes<Self::Shape, S0>> {
        for i in 0..D1 {
            for j in 0..D2 {
                for k in 0..D3 {
                    acc = f(acc, (self[i][j][k], *rhs))
                }
            }
        }

        Ok(acc)
    }
}

// S4 ---------------------------------------------------------------------- S4
// S4 ---------------------------------------------------------------------- S4
// S4 ---------------------------------------------------------------------- S4
impl<F, const D1: usize, const D2: usize, const D3: usize, const D4: usize>
    Class<S4<D1, D2, D3, D4>, F> for Arrays
where
    F: Scalar,
{
    type Buffer = [[[[F; D4]; D3]; D2]; D1];

    fn build(_: S4<D1, D2, D3, D4>, f: impl Fn([usize; 4]) -> F) -> [[[[F; D4]; D3]; D2]; D1] {
        array_init::array_init(|i| {
            array_init::array_init(|j| {
                array_init::array_init(|k| array_init::array_init(|u| f([i, j, k, u])))
            })
        })
    }

    fn build_subset(
        shape: S4<D1, D2, D3, D4>,
        base: F,
        indices: impl Iterator<Item = [usize; 4]>,
        active: impl Fn([usize; 4]) -> F,
    ) -> Self::Buffer {
        let mut buf = Self::full(shape, base);

        for [i1, i2, i3, i4] in indices {
            buf[i1][i2][i3][i4] = active([i1, i2, i3, i4]);
        }

        buf
    }

    fn full(_: S4<D1, D2, D3, D4>, value: F) -> Self::Buffer { [[[[value; D4]; D3]; D2]; D1] }
}

impl<F, const D1: usize, const D2: usize, const D3: usize, const D4: usize> Buffer
    for [[[[F; D4]; D3]; D2]; D1]
where
    F: Scalar,
{
    type Class = Arrays;
    type Field = F;
    type Shape = S4<D1, D2, D3, D4>;

    fn shape(&self) -> Self::Shape { S4 }

    fn get(&self, ix: [usize; 4]) -> Option<F> {
        if ix[0] < D1 && ix[1] < D2 && ix[2] < D3 && ix[3] < D4 {
            Some(self[ix[0]][ix[1]][ix[2]][ix[3]])
        } else {
            None
        }
    }

    fn map<M: Fn(F) -> F>(self, f: M) -> Self { Self::map(self, |x| Buffer::map(x, &f)) }

    fn map_ref<M: Fn(F) -> F>(&self, f: M) -> Self {
        array_init::array_init(|i| self[i].map_ref(&f))
    }

    fn fold<A, M: Fn(A, &F) -> A>(&self, mut init: A, f: M) -> A {
        for i in 0..D1 {
            init = self[i].fold(init, &f)
        }

        init
    }

    fn to_owned(&self) -> Self { self.clone() }

    fn into_owned(self) -> Self { self }
}

impl<F: Scalar, const D1: usize, const D2: usize, const D3: usize, const D4: usize> ZipMap
    for [[[[F; D4]; D3]; D2]; D1]
{
    fn zip_map(
        mut self,
        rhs: &Self,
        f: impl Fn(F, F) -> F,
    ) -> Result<Self, IncompatibleShapes<Self::Shape>> {
        for i in 0..D1 {
            self[i] = unsafe {
                self[i].zip_map(&rhs[i], &f).unwrap_unchecked()
            };
        }

        Ok(self)
    }

    fn zip_map_ref(
        &self,
        rhs: &Self,
        f: impl Fn(F, F) -> F,
    ) -> Result<Self, IncompatibleShapes<Self::Shape>> {
        Ok(array_init::array_init(|i| unsafe {
            self[i].zip_map(&rhs[i], &f).unwrap_unchecked()
        }))
    }

    fn take_left(lhs: Self) -> Self { lhs }

    fn take_right(rhs: Self) -> Self { rhs }
}

// impl<F: Scalar, const D1: usize, const D2: usize, const D3: usize, const D4: usize> ZipMap<F>
    // for [[[[F; D4]; D3]; D2]; D1]
// {
    // type Output = Self;

    // fn zip_map(
        // mut self,
        // rhs: &F,
        // f: impl Fn(F, F) -> F,
    // ) -> Result<Self, IncompatibleShapes<Self::Shape, S0>> {
        // for i in 0..D1 {
            // self[i] = unsafe {
                // self[i].zip_map(rhs, &f).unwrap_unchecked()
            // };
        // }

        // Ok(self)
    // }

    // fn zip_map_ref(
        // &self,
        // rhs: &F,
        // f: impl Fn(F, F) -> F,
    // ) -> Result<Self, IncompatibleShapes<Self::Shape, S0>> {
        // Ok(array_init::array_init(|i| unsafe {
            // self[i].zip_map(rhs, &f).unwrap_unchecked()
        // }))
    // }

    // fn take_left(lhs: Self, _: F::Shape) -> Result<Self, IncompatibleShapes<Self::Shape, S0>> { Ok(lhs) }

    // fn take_right(_: Self::Shape, rhs: F) -> Result<Self, IncompatibleShapes<Self::Shape, S0>> { Ok([[[[rhs; D4]; D3]; D2]; D1]) }
// }

impl<F: Scalar, const D1: usize, const D2: usize, const D3: usize, const D4: usize> ZipFold
    for [[[[F; D4]; D3]; D2]; D1]
{
    fn zip_fold(
        &self,
        rhs: &Self,
        mut acc: F,
        f: impl Fn(F, (F, F)) -> F,
    ) -> Result<F, IncompatibleShapes<Self::Shape>> {
        for i in 0..D1 {
            for j in 0..D2 {
                for k in 0..D3 {
                    for u in 0..D4 {
                        acc = f(acc, (self[i][j][k][u], rhs[i][j][k][u]))
                    }
                }
            }
        }

        Ok(acc)
    }
}

impl<F: Scalar, const D1: usize, const D2: usize, const D3: usize, const D4: usize> ZipFold<F>
    for [[[[F; D4]; D3]; D2]; D1]
{
    fn zip_fold(
        &self,
        rhs: &F,
        mut acc: F,
        f: impl Fn(F, (F, F)) -> F,
    ) -> Result<F, IncompatibleShapes<Self::Shape, S0>> {
        for i in 0..D1 {
            for j in 0..D2 {
                for k in 0..D3 {
                    for u in 0..D4 {
                        acc = f(acc, (self[i][j][k][u], *rhs))
                    }
                }
            }
        }

        Ok(acc)
    }
}

impl<F, const D1: usize, const D2: usize, const D3: usize, const D4: usize> Buffer
    for &[[[[F; D4]; D3]; D2]; D1]
where
    F: Scalar,
{
    type Class = Arrays;
    type Field = F;
    type Shape = S4<D1, D2, D3, D4>;

    fn shape(&self) -> Self::Shape { S4 }

    fn get(&self, ix: [usize; 4]) -> Option<F> {
        if ix[0] < D1 && ix[1] < D2 && ix[2] < D3 && ix[3] < D4 {
            Some(self[ix[0]][ix[1]][ix[2]][ix[3]])
        } else {
            None
        }
    }

    fn map<M: Fn(F) -> F>(self, f: M) -> [[[[F; D4]; D3]; D2]; D1] {
        <[[[[F; D4]; D3]; D2]; D1]>::map_ref(self, f)
    }

    fn map_ref<M: Fn(F) -> F>(&self, f: M) -> [[[[F; D4]; D3]; D2]; D1] {
        array_init::array_init(|i| self[i].map_ref(&f))
    }

    fn fold<A, M: Fn(A, &F) -> A>(&self, mut init: A, f: M) -> A {
        for i in 0..D1 {
            init = self[i].fold(init, &f)
        }

        init
    }

    fn to_owned(&self) -> [[[[F; D4]; D3]; D2]; D1] {
        <[[[[F; D4]; D3]; D2]; D1] as Buffer>::to_owned(self)
    }

    fn into_owned(self) -> [[[[F; D4]; D3]; D2]; D1] { self.clone() }
}

impl<F: Scalar, const D1: usize, const D2: usize, const D3: usize, const D4: usize> ZipMap
    for &[[[[F; D4]; D3]; D2]; D1]
{
    fn zip_map_ref(
        &self,
        rhs: &Self,
        f: impl Fn(F, F) -> F,
    ) -> Result<[[[[F; D4]; D3]; D2]; D1], IncompatibleShapes<Self::Shape>> {
        Ok(array_init::array_init(|i| unsafe {
            self[i].zip_map(&rhs[i], &f).unwrap_unchecked()
        }))
    }

    fn take_left(lhs: Self) -> [[[[F; D4]; D3]; D2]; D1] { lhs.into_owned() }

    fn take_right(rhs: Self) -> [[[[F; D4]; D3]; D2]; D1] { rhs.into_owned() }
}

// impl<F: Scalar, const D1: usize, const D2: usize, const D3: usize, const D4: usize> ZipMap<F>
    // for &[[[[F; D4]; D3]; D2]; D1]
// {
    // type Output = [[[[F; D4]; D3]; D2]; D1];

    // fn zip_map_ref(
        // &self,
        // rhs: &F,
        // f: impl Fn(F, F) -> F,
    // ) -> Result<Self::Output, IncompatibleShapes<Self::Shape, S0>> {
        // Ok(array_init::array_init(|i| unsafe {
            // self[i].zip_map(rhs, &f).unwrap_unchecked()
        // }))
    // }

    // fn take_left(lhs: Self, _: F::Shape) -> Result<Self::Output, IncompatibleShapes<Self::Shape, S0>> { Ok(lhs.into_owned()) }

    // fn take_right(_: Self::Shape, rhs: F) -> Result<Self::Output, IncompatibleShapes<Self::Shape, S0>> { Ok([[[[rhs; D4]; D3]; D2]; D1]) }
// }

impl<F: Scalar, const D1: usize, const D2: usize, const D3: usize, const D4: usize> ZipFold
    for &[[[[F; D4]; D3]; D2]; D1]
{
    fn zip_fold(
        &self,
        rhs: &Self,
        mut acc: F,
        f: impl Fn(F, (F, F)) -> F,
    ) -> Result<F, IncompatibleShapes<Self::Shape>> {
        for i in 0..D1 {
            for j in 0..D2 {
                for k in 0..D3 {
                    for u in 0..D4 {
                        acc = f(acc, (self[i][j][k][u], rhs[i][j][k][u]))
                    }
                }
            }
        }

        Ok(acc)
    }
}

impl<F: Scalar, const D1: usize, const D2: usize, const D3: usize, const D4: usize> ZipFold<F>
    for &[[[[F; D4]; D3]; D2]; D1]
{
    fn zip_fold(
        &self,
        rhs: &F,
        mut acc: F,
        f: impl Fn(F, (F, F)) -> F,
    ) -> Result<F, IncompatibleShapes<Self::Shape, S0>> {
        for i in 0..D1 {
            for j in 0..D2 {
                for k in 0..D3 {
                    for u in 0..D4 {
                        acc = f(acc, (self[i][j][k][u], *rhs))
                    }
                }
            }
        }

        Ok(acc)
    }
}

// S5 ---------------------------------------------------------------------- S5
// S5 ---------------------------------------------------------------------- S5
// S5 ---------------------------------------------------------------------- S5
impl<F, const D1: usize, const D2: usize, const D3: usize, const D4: usize, const D5: usize>
    Class<S5<D1, D2, D3, D4, D5>, F> for Arrays
where
    F: Scalar,
{
    type Buffer = [[[[[F; D5]; D4]; D3]; D2]; D1];

    fn build(
        _: S5<D1, D2, D3, D4, D5>,
        f: impl Fn([usize; 5]) -> F,
    ) -> [[[[[F; D5]; D4]; D3]; D2]; D1] {
        array_init::array_init(|i| {
            array_init::array_init(|j| {
                array_init::array_init(|k| {
                    array_init::array_init(|u| array_init::array_init(|l| f([i, j, k, u, l])))
                })
            })
        })
    }

    fn build_subset(
        shape: S5<D1, D2, D3, D4, D5>,
        base: F,
        indices: impl Iterator<Item = [usize; 5]>,
        active: impl Fn([usize; 5]) -> F,
    ) -> Self::Buffer {
        let mut buf = Self::full(shape, base);

        for [i1, i2, i3, i4, i5] in indices {
            buf[i1][i2][i3][i4][i5] = active([i1, i2, i3, i4, i5]);
        }

        buf
    }

    fn full(_: S5<D1, D2, D3, D4, D5>, value: F) -> Self::Buffer {
        [[[[[value; D5]; D4]; D3]; D2]; D1]
    }
}

impl<F, const D1: usize, const D2: usize, const D3: usize, const D4: usize, const D5: usize> Buffer
    for [[[[[F; D5]; D4]; D3]; D2]; D1]
where
    F: Scalar,
{
    type Class = Arrays;
    type Field = F;
    type Shape = S5<D1, D2, D3, D4, D5>;

    fn shape(&self) -> Self::Shape { S5 }

    fn get(&self, ix: [usize; 5]) -> Option<F> {
        if ix[0] < D1 && ix[1] < D2 && ix[2] < D3 && ix[3] < D4 && ix[4] < D5 {
            Some(self[ix[0]][ix[1]][ix[2]][ix[3]][ix[4]])
        } else {
            None
        }
    }

    fn map<M: Fn(F) -> F>(self, f: M) -> Self { Self::map(self, |x| Buffer::map(x, &f)) }

    fn map_ref<M: Fn(F) -> F>(&self, f: M) -> Self {
        array_init::array_init(|i| self[i].map_ref(&f))
    }

    fn fold<A, M: Fn(A, &F) -> A>(&self, mut init: A, f: M) -> A {
        for i in 0..D1 {
            init = self[i].fold(init, &f)
        }

        init
    }

    fn to_owned(&self) -> Self { self.clone() }

    fn into_owned(self) -> Self { self }
}

#[cfg(test)]
mod tests {
    use super::*;

    mod arr2 {
        use super::*;

        const V: [f64; 2] = [1.0, 2.0];

        #[test]
        fn test_ownership() {
            assert_eq!(Buffer::to_owned(&V), V);
            assert_eq!(Buffer::into_owned(V), V);
        }

        #[test]
        fn test_replace() {
            assert_eq!(V.to_zeroes(), [0.0; 2]);
            assert_eq!(V.into_zeroes(), [0.0; 2]);

            assert_eq!(V.to_ones(), [1.0; 2]);
            assert_eq!(V.into_ones(), [1.0; 2]);

            assert_eq!(V.to_filled(5.0), [5.0; 2]);
            assert_eq!(V.to_filled(-1.0), [-1.0; 2]);

            assert_eq!(V.into_filled(5.0), [5.0; 2]);
            assert_eq!(V.into_filled(-1.0), [-1.0; 2]);
        }

        #[test]
        fn test_transforms() {
            assert_eq!(V.map(|x| x * 2.0), [2.0, 4.0]);
            assert_eq!(V.fold(0.0, |a, x| a + x * 2.0), 6.0);
            assert_eq!(V.sum(), 3.0);
        }

        #[test]
        fn test_linalg() {
            assert_eq!(
                V.zip_fold(&V, 0.0, |acc, (xi, yi)| acc + xi * yi).unwrap(),
                5.0
            );
        }
    }
}
