use crate::{
    buffers::{Buffer, Compatible, FieldOf, IncompatibleShapes, ShapeOf, ZipFold},
    logic::TFU,
    ops::{Add, TensorMul},
    BinaryError,
    Contains,
    Database,
    Differentiable,
    Function,
    Identifier,
    Node,
    Stage,
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
#[derive(Copy, Clone, Debug, PartialEq, Contains)]
pub struct InnerProduct<N1, N2>(#[op] pub N1, #[op] pub N2);

impl<N1: Node, N2: Node> Node for InnerProduct<N1, N2> {
    fn is_zero(stage: Stage<&'_ Self>) -> TFU {
        stage.map(|node| &node.0).is_zero() | stage.map(|node| &node.1).is_zero()
    }

    fn is_one(stage: Stage<&'_ Self>) -> TFU {
        (stage.map(|node| &node.0).is_one() & stage.map(|node| &node.1).is_one())
            .true_or(TFU::Unknown)
    }
}

impl<D, N1, N2> Function<D> for InnerProduct<N1, N2>
where
    D: Database,
    N1: Function<D>,
    N2: Function<D>,

    N1::Value: Compatible<N2::Value> + ZipFold<N2::Value>,
{
    type Error = BinaryError<
        N1::Error,
        N2::Error,
        IncompatibleShapes<ShapeOf<N1::Value>, ShapeOf<N2::Value>>,
    >;
    type Value = FieldOf<N1::Value>;

    fn evaluate<DR: AsRef<D>>(&self, db: DR) -> Result<Self::Value, Self::Error> {
        use crate::Stage::Evaluation;

        let s0 = Evaluation(&self.0);
        let s1 = Evaluation(&self.1);

        // If either value is zero, we can just short-circuit...
        if (s0.is_zero() | s1.is_zero()).is_true() {
            return Ok(num_traits::zero());
        }

        // If either value is unity, then we can also short-circuit...
        if s0.is_one().is_true() {
            // In this case, the left-hand value is one, i.e. we have 1 * x = x.
            return self
                .0
                .evaluate(db.as_ref())
                .map_err(BinaryError::Left)
                .map(|buf| buf.sum());
        }

        if s1.is_one().is_true() {
            // In this case, the right-hand value is one, i.e. we have x * 1 = x.
            return self
                .1
                .evaluate(db.as_ref())
                .map_err(BinaryError::Right)
                .map(|buf| buf.sum());
        }

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
