use crate::{
    buffers::{Buffer, FieldOf, IncompatibleShapes, ShapeOf},
    ops::Add,
    BinaryError,
    Contains,
    Database,
    Differentiable,
    Function,
    Identifier,
    Node,
};

pub trait TensorMulTrait<T>: Buffer
where
    T: Buffer<Field = Self::Field>,
{
    type Output: Buffer<Field = Self::Field>;

    fn tensor_mul(
        &self,
        rhs: &T,
    ) -> Result<Self::Output, IncompatibleShapes<Self::Shape, T::Shape>>;
}

mod impls;

/// Computes the product of two matrix [Buffers](Buffer).
///
/// # Examples
/// ```
/// # #[macro_use] extern crate aegir;
/// # use aegir::{Identifier, Function, Differentiable, Dual, ops::TensorMul, ids::{X, Y}};
/// db!(DB { x: X, y: Y });
///
/// let f = TensorMul(X.into_var(), Y.into_var());
/// let db = DB {
///     x: [
///         [1.0, 2.0],
///         [3.0, 4.0]
///     ],
///     y: [
///         [4.0, 3.0],
///         [2.0, 1.0]
///     ]
/// };
///
/// assert_eq!(f.evaluate(&db).unwrap(), [
///     [8.0, 5.0],
///     [20.0, 13.0]
/// ]);
/// assert_eq!(f.evaluate_adjoint(X, &db).unwrap(), [[
///     [[4., 3.],
///      [0., 0.]],
///
///     [[2., 1.],
///      [0., 0.]]
/// ], [
///     [[0., 0.],
///      [4., 3.]],
///
///     [[0., 0.],
///      [2., 1.]]
/// ]]);
/// ```
#[derive(Clone, Debug, PartialEq, Node, Contains)]
pub struct TensorMul<N1, N2>(#[op] pub N1, #[op] pub N2);

impl<D, N1, N2> Function<D> for TensorMul<N1, N2>
where
    D: Database,

    N1: Function<D>,
    N2: Function<D>,

    N1::Value: TensorMulTrait<N2::Value>,
    N2::Value: Buffer<Field = FieldOf<N1::Value>>,
{
    type Error = BinaryError<
        N1::Error,
        N2::Error,
        IncompatibleShapes<ShapeOf<N1::Value>, ShapeOf<N2::Value>>,
    >;
    type Value = <N1::Value as TensorMulTrait<N2::Value>>::Output;

    fn evaluate<DR: AsRef<D>>(&self, db: DR) -> Result<Self::Value, Self::Error> {
        let x = self.0.evaluate(db.as_ref()).map_err(BinaryError::Left)?;
        let y = self.1.evaluate(db).map_err(BinaryError::Right)?;

        x.tensor_mul(&y).map_err(BinaryError::Output)
    }
}

impl<T, N1, N2> Differentiable<T> for TensorMul<N1, N2>
where
    T: Identifier,
    N1: Differentiable<T> + Clone,
    N2: Differentiable<T> + Clone,
{
    type Adjoint = Add<TensorMul<N1::Adjoint, N2>, TensorMul<N1, N2::Adjoint>>;

    fn adjoint(&self, target: T) -> Self::Adjoint {
        let l = TensorMul(self.0.adjoint(target), self.1.clone());
        let r = TensorMul(self.0.clone(), self.1.adjoint(target));

        Add(l, r)
    }
}

impl<N1, N2> std::fmt::Display for TensorMul<N1, N2>
where
    N1: std::fmt::Display,
    N2: std::fmt::Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({}) ({})", self.0, self.1)
    }
}
