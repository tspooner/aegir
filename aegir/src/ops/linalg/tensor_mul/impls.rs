use super::TensorMulTrait;
use crate::buffers::{shapes, Buffer, IncompatibleShapes, OwnedOf, Scalar, ShapeOf};

// Base Cases --------------------------------------
impl<F, B> TensorMulTrait<B> for F
where
    F: Scalar,
    B: Buffer<Field = F>,
{
    type Output = OwnedOf<B>;

    #[inline]
    fn tensor_mul(
        &self,
        rhs: &B,
    ) -> Result<Self::Output, IncompatibleShapes<shapes::S0, B::Shape>> {
        Ok(rhs.map_ref(|x| *self * x))
    }
}

impl<F, const N: usize, const M: usize> TensorMulTrait<[F; M]> for [[F; M]; N]
where
    F: Scalar,
{
    type Output = [[F; N]; 1];

    #[inline]
    fn tensor_mul(
        &self,
        rhs: &[F; M],
    ) -> Result<Self::Output, IncompatibleShapes<shapes::S2<N, M>, shapes::S1<M>>> {
        let out =
            array_init::array_init(|i| (0..M).fold(F::zero(), |acc, k| acc + self[i][k] * rhs[k]));

        Ok([out])
    }
}

impl<F, const N: usize, const Z: usize, const M: usize> TensorMulTrait<[[F; M]; Z]> for [[F; Z]; N]
where
    F: Scalar,
{
    type Output = [[F; M]; N];

    #[inline]
    fn tensor_mul(
        &self,
        rhs: &[[F; M]; Z],
    ) -> Result<Self::Output, IncompatibleShapes<shapes::S2<N, Z>, shapes::S2<Z, M>>> {
        let out = array_init::array_init(|i| {
            array_init::array_init(|j| {
                (0..Z).fold(F::zero(), |acc, z| acc + self[i][z] * rhs[z][j])
            })
        });

        Ok(out)
    }
}

impl<F, const N: usize, const Z: usize, const M1: usize, const M2: usize>
    TensorMulTrait<[[[F; M2]; M1]; Z]> for [[F; Z]; N]
where
    F: Scalar,
{
    type Output = [[[F; M2]; M1]; N];

    #[inline]
    fn tensor_mul(
        &self,
        rhs: &[[[F; M2]; M1]; Z],
    ) -> Result<Self::Output, IncompatibleShapes<shapes::S2<N, Z>, shapes::S3<Z, M1, M2>>> {
        let out = array_init::array_init(|i| {
            array_init::array_init(|j| {
                array_init::array_init(|k| {
                    (0..Z).fold(F::zero(), |acc, z| acc + self[i][z] * rhs[z][j][k])
                })
            })
        });

        Ok(out)
    }
}

impl<F, const N: usize, const Z: usize, const M1: usize, const M2: usize, const M3: usize>
    TensorMulTrait<[[[[F; M3]; M2]; M1]; Z]> for [[F; Z]; N]
where
    F: Scalar,
{
    type Output = [[[[F; M3]; M2]; M1]; N];

    #[inline]
    fn tensor_mul(
        &self,
        rhs: &[[[[F; M3]; M2]; M1]; Z],
    ) -> Result<Self::Output, IncompatibleShapes<shapes::S2<N, Z>, shapes::S4<Z, M1, M2, M3>>>
    {
        let out = array_init::array_init(|i| {
            array_init::array_init(|j| {
                array_init::array_init(|k| {
                    array_init::array_init(|l| {
                        (0..Z).fold(F::zero(), |acc, z| acc + self[i][z] * rhs[z][j][k][l])
                    })
                })
            })
        });

        Ok(out)
    }
}

// Recursions --------------------------------------
impl<F, B, const N1: usize, const N2: usize, const N3: usize> TensorMulTrait<B>
    for [[[F; N3]; N2]; N1]
where
    F: Scalar,
    B: Buffer<Field = F>,

    [[F; N3]; N2]: Buffer<Field = F> + TensorMulTrait<B>,
    [<[[F; N3]; N2] as TensorMulTrait<B>>::Output; N1]: Buffer<Field = F>,
{
    type Output = [<[[F; N3]; N2] as TensorMulTrait<B>>::Output; N1];

    #[inline]
    fn tensor_mul(
        &self,
        rhs: &B,
    ) -> Result<Self::Output, IncompatibleShapes<ShapeOf<Self>, ShapeOf<B>>> {
        let out = array_init::array_init(|i| self[i].tensor_mul(rhs).unwrap());

        Ok(out)
    }
}

impl<F, B, const N1: usize, const N2: usize, const N3: usize, const N4: usize> TensorMulTrait<B>
    for [[[[F; N4]; N3]; N2]; N1]
where
    F: Scalar,
    B: Buffer<Field = F>,

    [[F; N4]; N3]: Buffer<Field = F> + TensorMulTrait<B>,
    [[<[[F; N4]; N3] as TensorMulTrait<B>>::Output; N2]; N1]: Buffer<Field = F>,
{
    type Output = [[<[[F; N4]; N3] as TensorMulTrait<B>>::Output; N2]; N1];

    #[inline]
    fn tensor_mul(
        &self,
        rhs: &B,
    ) -> Result<Self::Output, IncompatibleShapes<ShapeOf<Self>, ShapeOf<B>>> {
        let out = array_init::array_init(|i| {
            array_init::array_init(|j| self[i][j].tensor_mul(rhs).unwrap())
        });

        Ok(out)
    }
}
