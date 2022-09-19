use crate::{
    buffers::{Compatible, FieldOf, IncompatibleShapes, ShapeOf, ZipFold},
    ops::{Add, TensorMul},
    BinaryError,
    Contains,
    Database,
    Differentiable,
    Function,
    Identifier,
    Node,
};

/// Computes the inner product of two vector [Buffers](Buffer).
///
/// # Examples
/// ```
/// # #[macro_use] extern crate aegir;
/// # use aegir::{Identifier, Differentiable, Dual, ops::InnerProduct, ids::{X, Y}};
/// db!(DB { x: X, y: Y });
///
/// let f = InnerProduct(X.into_var(), Y.into_var());
/// let db = DB {
///     x: [1.0, 2.0, 3.0],
///     y: [-1.0, 0.0, 2.0]
/// };
///
/// assert_eq!(f.evaluate_dual(X, &db).unwrap(), dual!(5.0, [[-1.0, 0.0, 2.0]]));
/// assert_eq!(f.evaluate_dual(Y, &db).unwrap(), dual!(5.0, [[1.0, 2.0, 3.0]]));
/// ```
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct InnerProduct<N1, N2>(pub N1, pub N2);

impl<N1, N2> Node for InnerProduct<N1, N2> {}

impl<T, N1, N2> Contains<T> for InnerProduct<N1, N2>
where
    T: Identifier,
    N1: Contains<T>,
    N2: Contains<T>,
{
    fn contains(&self, target: T) -> bool { self.0.contains(target) || self.1.contains(target) }
}

impl<D, N1, N2> Function<D> for InnerProduct<N1, N2>
where
    D: Database,
    N1: Function<D>,
    N2: Function<D>,

    N1::Value: Compatible<N2::Value>,
{
    type Error = BinaryError<
        N1::Error,
        N2::Error,
        IncompatibleShapes<ShapeOf<N1::Value>, ShapeOf<N2::Value>>,
    >;
    type Value = FieldOf<N1::Value>;

    fn evaluate<DR: AsRef<D>>(&self, db: DR) -> Result<Self::Value, Self::Error> {
        let x = self.0.evaluate(db.as_ref()).map_err(BinaryError::Left)?;
        let y = self.1.evaluate(db).map_err(BinaryError::Right)?;

        x.zip_fold(&y, num_traits::zero(), |acc, (xi, yi)| acc + xi * yi)
            .map_err(BinaryError::Output)
    }
}

impl<T, N1, N2> Differentiable<T> for InnerProduct<N1, N2>
where
    T: Identifier,
    N1: Differentiable<T> + Clone,
    N2: Differentiable<T> + Clone,
{
    type Adjoint = Add<TensorMul<N2::Adjoint, N1>, TensorMul<N1::Adjoint, N2>>;

    fn adjoint(&self, target: T) -> Self::Adjoint {
        let j1 = self.0.adjoint(target);
        let j2 = self.1.adjoint(target);

        Add(TensorMul(j2, self.0.clone()), TensorMul(j1, self.1.clone()))
    }
}

impl<N1: std::fmt::Display, N2: std::fmt::Display> std::fmt::Display for InnerProduct<N1, N2> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "\u{27E8}{}, {}\u{27E9}", self.0, self.1)
    }
}
