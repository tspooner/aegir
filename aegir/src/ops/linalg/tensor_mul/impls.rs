use super::TensorMulTrait;
use crate::buffers::{shapes::{self, Concat, Split, Shape}, Class, Buffer, BufferOf, IncompatibleShapes, Scalars, Arrays, OwnedOf, Scalar, ShapeOf, precedence::{Precedence, PBufferOf}};

// Base Cases --------------------------------------
// 1 . X
impl<F, B> TensorMulTrait<B> for F
where
    F: Scalar,
    B: Buffer<Field = F>,

    Scalars: Precedence<B::Class, B::Shape, F, Class = B::Class>,
{
    type OutShape = B::Shape;

    #[inline]
    fn tensor_mul(
        &self,
        rhs: &B,
    ) -> Result<OwnedOf<B>, IncompatibleShapes<F::Shape, B::Shape>> {
        Ok(rhs.map_ref(|x| *self * x))
    }

    fn tensor_mul_s(
        _: F::Shape,
        shape_right: B::Shape,
    ) -> Result<B::Shape, IncompatibleShapes<F::Shape, B::Shape>> {
        Ok(shape_right)
    }
}

// N x M . M x 1 =>  1 x N
impl<F, const N: usize, const M: usize> TensorMulTrait<[F; M]> for [[F; M]; N]
where
    F: Scalar,
{
    type OutShape = shapes::S2<N, 1>;

    #[inline]
    fn tensor_mul(
        &self,
        rhs: &[F; M],
    ) -> Result<[[F; 1]; N], IncompatibleShapes<shapes::S2<N, M>, shapes::S1<M>>> {
        let out =
            array_init::array_init(|i| [(0..M).fold(F::zero(), |acc, k| acc + self[i][k] * rhs[k])]);

        Ok(out)
    }

    fn tensor_mul_s(
        _: shapes::S2<N, M>,
        _: shapes::S1<M>,
    ) -> Result<Self::OutShape, IncompatibleShapes<shapes::S2<N, M>, shapes::S1<M>>>
    {
        Ok(shapes::S2)
    }
}

impl<F, const N: usize, const Z: usize, const M: usize> TensorMulTrait<[[F; M]; Z]> for [[F; Z]; N]
where
    F: Scalar,
{
    type OutShape = shapes::S2<M, N>;

    #[inline]
    fn tensor_mul(
        &self,
        rhs: &[[F; M]; Z],
    ) -> Result<[[F; N]; M], IncompatibleShapes<shapes::S2<N, Z>, shapes::S2<Z, M>>> {
        let out = array_init::array_init(|i| {
            array_init::array_init(|j| {
                (0..Z).fold(F::zero(), |acc, z| acc + self[i][z] * rhs[z][j])
            })
        });

        Ok(out)
    }

    fn tensor_mul_s(
        _: shapes::S2<N, Z>,
        _: shapes::S2<Z, M>,
    ) -> Result<Self::OutShape, IncompatibleShapes<shapes::S2<N, Z>, shapes::S2<Z, M>>>
    {
        Ok(shapes::S2)
    }
}

impl<F, const N: usize, const Z: usize, const M1: usize, const M2: usize>
    TensorMulTrait<[[[F; M2]; M1]; Z]> for [[F; Z]; N]
where
    F: Scalar,
{
    type OutShape = shapes::S3<N, M1, M2>;

    #[inline]
    fn tensor_mul(
        &self,
        rhs: &[[[F; M2]; M1]; Z],
    ) -> Result<[[[F; M2]; M1]; N], IncompatibleShapes<shapes::S2<N, Z>, shapes::S3<Z, M1, M2>>> {
        let out = array_init::array_init(|i| {
            array_init::array_init(|j| {
                array_init::array_init(|k| {
                    (0..Z).fold(F::zero(), |acc, z| acc + self[i][z] * rhs[z][j][k])
                })
            })
        });

        Ok(out)
    }

    fn tensor_mul_s(
        _: shapes::S2<N, Z>,
        _: shapes::S3<Z, M1, M2>,
    ) -> Result<Self::OutShape, IncompatibleShapes<shapes::S2<N, Z>, shapes::S3<Z, M1, M2>>>
    {
        Ok(shapes::S3)
    }
}

impl<F, const N: usize, const Z: usize, const M1: usize, const M2: usize, const M3: usize>
    TensorMulTrait<[[[[F; M3]; M2]; M1]; Z]> for [[F; Z]; N]
where
    F: Scalar,
{
    type OutShape = shapes::S4<Z, M1, M2, M3>;

    #[inline]
    fn tensor_mul(
        &self,
        rhs: &[[[[F; M3]; M2]; M1]; Z],
    ) -> Result<[[[[F; M3]; M2]; M1]; Z], IncompatibleShapes<shapes::S2<N, Z>, shapes::S4<Z, M1, M2, M3>>>
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

    fn tensor_mul_s(
        _: shapes::S2<N, Z>,
        _: shapes::S4<Z, M1, M2, M3>,
    ) -> Result<Self::OutShape, IncompatibleShapes<shapes::S2<N, Z>, shapes::S4<Z, M1, M2, M3>>>
    {
        Ok(shapes::S4)
    }
}

// Recursions --------------------------------------
    // [[F; N3]; N2]: Buffer<Field = F, Shape = shapes::S2<N2, N3>> + TensorMulTrait<B>,
    // [<[[F; N3]; N2] as TensorMulTrait<B>>::Output; N1]: Buffer<
        // Field = F,
        // Shape = <shapes::S1<N1> as Concat<ShapeOf<<[[F; N3]; N2] as TensorMulTrait<B>>::Output>>>::Shape
    // >,

// impl<F, B, SB, SI, SO, const N1: usize, const N2: usize, const N3: usize> TensorMulTrait<B>
    // for [[[F; N3]; N2]; N1]
// where
    // F: Scalar,
    // B: Buffer<Class = Arrays, Field = F, Shape = SB>,

    // SB: Shape,
    // SI: Shape,
    // SO: Shape,

    // Arrays: Class<SB, F> + Class<SI, F> + Class<SO, F> + Precedence<Arrays, SI, F, Class = Arrays> + Precedence<Arrays, SO, F, Class = Arrays>,

    // [[F; N3]; N2]: Buffer<Class = Arrays, Field = F> + TensorMulTrait<B, OutShape = SI>,

    // [BufferOf<Arrays, SI, F>; N1]: Buffer<Class = Arrays, Field = F, Shape = SO>,

    // shapes::S1<N1>: Concat<SI, Shape = SO>,
// {
    // type OutShape = SO;

    // #[inline]
    // fn tensor_mul(
        // &self,
        // rhs: &B,
    // ) -> Result<
        // [BufferOf<Arrays, SI, F>; N1],
        // IncompatibleShapes<shapes::S3<N1, N2, N3>, B::Shape>
    // > {
        // let out = array_init::array_init(|i| self[i].tensor_mul(rhs).unwrap());

        // Ok(out)
    // }

    // fn tensor_mul_s(
        // _: shapes::S3<N1, N2, N3>,
        // shape_right: B::Shape,
    // ) -> Result<SO, IncompatibleShapes<shapes::S3<N1, N2, N3>, B::Shape>>
    // {
        // unimplemented!()
        // // let inner = <[[F; N3]; N2] as TensorMulTrait<B>>::tensor_mul_s(shapes::S2, shape_right)
            // // .map_err(|_| IncompatibleShapes(shapes::S3, shape_right))?;

        // // Ok(shapes::S1.concat(inner))
    // }
// }

// // impl<F, B, const N1: usize, const N2: usize, const N3: usize, const N4: usize> TensorMulTrait<B>
    // // for [[[[F; N4]; N3]; N2]; N1]
// // where
    // // F: Scalar,
    // // B: Buffer<Field = F>,

    // // [[F; N4]; N3]: Buffer<Field = F, Shape = shapes::S2<N3, N4>> + TensorMulTrait<B>,
    // // [[<[[F; N4]; N3] as TensorMulTrait<B>>::Output; N2]; N1]: Buffer<
        // // Field = F,
        // // Shape = <shapes::S2<N1, N2> as Concat<ShapeOf<<[[F; N4]; N3] as TensorMulTrait<B>>::Output>>>::Shape
    // // >,

    // // shapes::S2<N1, N2>: Concat<ShapeOf<<[[F; N4]; N3] as TensorMulTrait<B>>::Output>>,
// // {
    // // type Output = [[<[[F; N4]; N3] as TensorMulTrait<B>>::Output; N2]; N1];

    // // #[inline]
    // // fn tensor_mul(
        // // &self,
        // // rhs: &B,
    // // ) -> Result<Self::Output, IncompatibleShapes<ShapeOf<Self>, ShapeOf<B>>> {
        // // let out = array_init::array_init(|i| {
            // // array_init::array_init(|j| self[i][j].tensor_mul(rhs).unwrap())
        // // });

        // // Ok(out)
    // // }

    // // fn tensor_mul_s(
        // // _: shapes::S4<N1, N2, N3, N4>,
        // // shape_right: B::Shape,
    // // ) -> Result<ShapeOf<Self::Output>, IncompatibleShapes<shapes::S4<N1, N2, N3, N4>, B::Shape>>
    // // {
        // // let inner = <[[F; N4]; N3] as TensorMulTrait<B>>::tensor_mul_s(shapes::S2, shape_right)
            // // .map_err(|_| IncompatibleShapes(shapes::S4, shape_right))?;

        // // Ok(shapes::S2.concat(inner))
    // // }
// // }
