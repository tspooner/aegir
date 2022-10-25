use crate::{
    buffers::{Buffer, Scalar, BufferOf, FieldOf, IncompatibleShapes, ShapeOf, Class, shapes::{Shape, Concat}, precedence::{Precedence, PBufferOf}},
    logic::TFU,
    ops::Add,
    BinaryError,
    Contains,
    Database,
    Differentiable,
    Function,
    Identifier,
    Node,
    Stage,
};

pub trait TensorMulTrait<T>: Buffer
where
    T: Buffer<Field = Self::Field>,

    Self::Class: Precedence<T::Class, Self::OutShape, Self::Field>
{
    type OutShape: Shape;

    fn tensor_mul(
        &self,
        rhs: &T,
    ) -> Result<
        PBufferOf<Self::Class, T::Class, Self::OutShape, Self::Field>,
        IncompatibleShapes<Self::Shape, T::Shape>
    >;

    fn tensor_mul_s(
        shape_left: Self::Shape,
        shape_right: T::Shape,
    ) -> Result<Self::OutShape, IncompatibleShapes<Self::Shape, T::Shape>>;
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
#[derive(Clone, Debug, PartialEq, Contains)]
pub struct TensorMul<N1, N2>(#[op] pub N1, #[op] pub N2);

impl<N1: Node, N2: Node> Node for TensorMul<N1, N2> {
    fn is_zero(stage: Stage<&'_ Self>) -> TFU {
        stage.map(|node| &node.0).is_zero() | stage.map(|node| &node.1).is_zero()
    }

    fn is_one(stage: Stage<&'_ Self>) -> TFU {
        (stage.map(|node| &node.0).is_one() & stage.map(|node| &node.1).is_one())
            .true_or(TFU::Unknown)
    }
}

impl<D, F, S1, S2, SO, C1, C2, B1, B2, N1, N2> Function<D> for TensorMul<N1, N2>
where
    D: Database,
    F: Scalar,

    S1: Shape,
    S2: Shape,
    SO: Shape,

    C1: Class<S1, F> + Precedence<C2, SO, F>,
    C2: Class<S2, F>,

    B1: Buffer<Class = C1, Shape = S1, Field = F> + TensorMulTrait<B2, OutShape = SO>,
    B2: Buffer<Class = C2, Shape = S2, Field = F>,

    N1: Function<D, Value = B1>,
    N2: Function<D, Value = B2>,
{
    type Error = BinaryError<
        N1::Error, N2::Error,
        IncompatibleShapes<S1, S2>,
    >;
    type Value = PBufferOf<C1, C2, SO, F>;

    fn evaluate<DR: AsRef<D>>(&self, db: DR) -> Result<Self::Value, Self::Error> {
        use crate::Stage::Evaluation;

        let s0 = Evaluation(&self.0);
        let s1 = Evaluation(&self.1);

        // If either value is zero, we can just short-circuit...
        if (s0.is_zero() | s1.is_zero()).is_true() {
            // TODO XXX - Replace calls to unwrap!
            let out_shape = <N1::Value as TensorMulTrait<N2::Value>>::tensor_mul_s(
                self.0.evaluate_shape(db.as_ref()).map_err(BinaryError::Left)?,
                self.1.evaluate_shape(db).map_err(BinaryError::Right)?
            ).unwrap();

            return Ok(<Self::Value as Buffer>::zeroes(out_shape));
        }

        // // If either value is unity, then we can also short-circuit...
        // if s0.is_one().is_true() {
            // // In this case, the left-hand value is one, i.e. we have 1 * x = x.
            // return self
                // .0
                // .evaluate(db.as_ref())
                // .map_err(BinaryError::Left)
                // .map(<N1::Value as ZipMap<N2::Value>>::take_left);
        // }

        // if s1.is_one().is_true() {
            // // In this case, the right-hand value is one, i.e. we have x * 1 = x.
            // return self
                // .1
                // .evaluate(db.as_ref())
                // .map_err(BinaryError::Right)
                // .map(<N1::Value as ZipMap<N2::Value>>::take_right);
        // }

        // Otherwise we have to perform the multiplication...
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
    N1: Node + std::fmt::Display,
    N2: Node + std::fmt::Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use crate::Stage::Instance;

        if Instance(&self.0).is_zero() == TFU::True {
            write!(f, "{}", self.1)
        } else if Instance(&self.1).is_zero() == TFU::True {
            write!(f, "{}", self.0)
        } else {
            write!(f, "({}) ({})", self.0, self.1)
        }
    }
}
