use crate::{
    buffers::{shapes, Buffer, FieldOf, IncompatibleShapes, Scalar, ShapeOf},
    logic::TFU,
    ops::Mul,
    BinaryError,
    Contains,
    Database,
    Differentiable,
    Function,
    Identifier,
    Node,
    Stage,
};

pub trait OuterProductTrait<T>: Buffer
where
    T: Buffer<Field = Self::Field>,

    Self::Shape: shapes::Concat<T::Shape>,
{
    type Output: Buffer<
        Shape = <Self::Shape as shapes::Concat<T::Shape>>::Shape,
        Field = Self::Field,
    >;

    fn outer_prod(
        &self,
        rhs: &T,
    ) -> Result<Self::Output, IncompatibleShapes<Self::Shape, T::Shape>>;
}

impl<F, const N: usize, const M: usize> OuterProductTrait<[F; M]> for [F; N]
where
    F: Scalar,
{
    type Output = [[F; M]; N];

    fn outer_prod(
        &self,
        rhs: &[F; M],
    ) -> Result<[[F; M]; N], IncompatibleShapes<shapes::S1<N>, shapes::S1<M>>> {
        let out = array_init::array_init(|i| array_init::array_init(|j| self[i] * rhs[j]));

        Ok(out)
    }
}

impl<F, const N: usize, const M: usize, const P: usize> OuterProductTrait<[[F; M]; P]> for [F; N]
where
    F: Scalar,
{
    type Output = [[[F; M]; P]; N];

    fn outer_prod(
        &self,
        rhs: &[[F; M]; P],
    ) -> Result<Self::Output, IncompatibleShapes<shapes::S1<N>, shapes::S2<P, M>>> {
        let out = array_init::array_init(|i| {
            array_init::array_init(|j| array_init::array_init(|k| self[i] * rhs[j][k]))
        });

        Ok(out)
    }
}

impl<F, const N: usize, const M: usize, const P: usize> OuterProductTrait<[F; P]> for [[F; M]; N]
where
    F: Scalar,
{
    type Output = [[[F; P]; M]; N];

    fn outer_prod(
        &self,
        rhs: &[F; P],
    ) -> Result<Self::Output, IncompatibleShapes<shapes::S2<N, M>, shapes::S1<P>>> {
        let out = array_init::array_init(|i| {
            array_init::array_init(|j| array_init::array_init(|k| self[i][j] * rhs[k]))
        });

        Ok(out)
    }
}

// impl<F, D> OuterProductTrait<ArrayBase<D, Ix1>> for ArrayBase<D, Ix1>
// where
// F: Scalar + 'static,
// D: ndarray::Data<Elem = F> + ndarray::RawDataClone,
// {
// type Output = ndarray::Array2<F>;

// fn outer_product<E1, E2>(
// &self,
// rhs: &ArrayBase<D, Ix1>,
// ) -> Result<ndarray::Array2<F>, LinalgError<E1, E2>> {
// let nx = self.len();
// let ny = rhs.len();

// self.view()
// .into_shape((nx, 1))
// .unwrap()
// .mat_mul(&rhs.view().into_shape((1, ny)).unwrap())
// }
// }

// impl<F> OuterProductTrait<Vec<F>> for Vec<F>
// where
// F: Scalar + 'static,
// {
// type Output = ndarray::Array2<F>;

// fn outer_product<E1, E2>(
// &self,
// rhs: &Vec<F>,
// ) -> Result<ndarray::Array2<F>, LinalgError<E1, E2>> {
// ndarray::aview1(self).outer_product(&ndarray::aview1(rhs))
// }
// }

/// Computes the outer product of two vector [Buffers](Buffer).
///
/// # Examples
/// ```
/// # #[macro_use] extern crate aegir;
/// # use aegir::{Identifier, Function, Differentiable, Dual, ops::OuterProduct, ids::{X, Y}};
/// db!(DB { x: X, y: Y });
///
/// let f = OuterProduct(X.into_var(), Y.into_var());
/// let db = DB {
///     x: [1.0, 2.0, 3.0],
///     y: [-1.0, 0.0, 2.0]
/// };
///
/// assert_eq!(f.evaluate(&db).unwrap(), [
///     [-1.0, 0.0, 2.0],
///     [-2.0, 0.0, 4.0],
///     [-3.0, 0.0, 6.0]
/// ]);
// /// assert_eq!(f.evaluate_grad(&db, X).unwrap(), array![
// ///     [-1.0, 0.0, 2.0],
// ///     [-1.0, 0.0, 2.0],
// ///     [-1.0, 0.0, 2.0]
// /// ]);
// /// assert_eq!(f.evaluate_grad(&db, Y).unwrap(), array![
// ///     [1.0, 1.0, 1.0],
// ///     [2.0, 2.0, 2.0],
// ///     [3.0, 3.0, 3.0]
// /// ]);
/// ```
#[derive(Copy, Clone, Debug, PartialEq, Contains)]
pub struct OuterProduct<N1, N2>(#[op] pub N1, #[op] pub N2);

impl<N1: Node, N2: Node> Node for OuterProduct<N1, N2> {
    fn is_zero(stage: Stage<&'_ Self>) -> TFU {
        stage.map(|node| &node.0).is_zero() | stage.map(|node| &node.1).is_zero()
    }

    fn is_one(stage: Stage<&'_ Self>) -> TFU {
        (stage.map(|node| &node.0).is_one() & stage.map(|node| &node.1).is_one())
            .true_or(TFU::Unknown)
    }
}

impl<D, N1, N2> Function<D> for OuterProduct<N1, N2>
where
    D: Database,
    N1: Function<D>,
    N2: Function<D>,

    N1::Value: OuterProductTrait<N2::Value>,
    N2::Value: Buffer<Field = FieldOf<N1::Value>>,

    ShapeOf<N1::Value>: shapes::Concat<ShapeOf<N2::Value>>,
{
    type Error = BinaryError<
        N1::Error,
        N2::Error,
        IncompatibleShapes<ShapeOf<N1::Value>, ShapeOf<N2::Value>>,
    >;
    type Value = <N1::Value as OuterProductTrait<N2::Value>>::Output;

    fn evaluate<DR: AsRef<D>>(&self, db: DR) -> Result<Self::Value, Self::Error> {
        let x = self.0.evaluate(db.as_ref()).map_err(BinaryError::Left)?;
        let y = self.1.evaluate(db).map_err(BinaryError::Right)?;

        x.outer_prod(&y).map_err(BinaryError::Output)
    }
}

// impl<D, T, N1, N2> Differentiable<D, T> for OuterProduct<N1, N2>
// where
// D: Database,
// T: Identifier,
// N1: Differentiable<D, T>,
// N2: Differentiable<D, T>,

// N1::Value: OuterProductTrait<N2::Value>,
// N2::Value: Buffer<Field = FieldOf<N1::Value>>,

// N1::Jacobian: OuterProductTrait<N2::Value>,
// N1::Value: OuterProductTrait<N2::Jacobian>,

// <N1::Jacobian as OuterProductTrait<N2::Value>>::Output:
// std::ops::Add<<N1::Value as OuterProductTrait<N2::Jacobian>>::Output>,

// AddOut<
// <N1::Jacobian as OuterProductTrait<N2::Value>>::Output,
// <N1::Value as OuterProductTrait<N2::Jacobian>>::Output,
// >: Buffer<Field = FieldOf<Self::Value>>,
// {
// type Jacobian = AddOut<
// <N1::Jacobian as OuterProductTrait<N2::Value>>::Output,
// <N1::Value as OuterProductTrait<N2::Jacobian>>::Output,
// >;

// fn grad(&self, db: &D, target: T) -> Result<Self::Jacobian, Self::Error> {
// let dual_x = self
// .0
// .dual(db, target)
// .map_err(|e| LinalgError::Inner1(e))?;
// let dual_y = self
// .1
// .dual(db, target)
// .map_err(|e| LinalgError::Inner2(e))?;

// dual_x
// .adjoint
// .outer_product(&dual_y.value)
// .and_then(|x| dual_x.value.outer_product(&dual_y.adjoint).map(|y| x + y))
// }
// }

impl<N1, N2> std::fmt::Display for OuterProduct<N1, N2>
where
    N1: std::fmt::Display,
    N2: std::fmt::Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} \u{2297} {}", self.0, self.1)
    }
}
