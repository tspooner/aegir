use crate::{
    buffers::{Buffer, FieldOf, IncompatibleShapes, ShapeOf, Contract as CTrait},
    ops::Add,
    BinaryError,
    Contains,
    Database,
    Differentiable,
    Function,
    Identifier,
    Node,
};

#[derive(Copy, Clone, PartialEq, Contains)]
pub struct Contract<const AXES: usize, N1, N2>(#[op] pub N1, #[op] pub N2);

impl<N1, N2, const AXES: usize> Contract<AXES, N1, N2> {
    pub fn new(left: N1, right: N2) -> Self { Contract(left, right) }
}

impl<N1, N2, const AXES: usize> Node for Contract<AXES, N1, N2> {}

impl<D, N1, N2, const AXES: usize> Function<D> for Contract<AXES, N1, N2>
where
    D: Database,
    N1: Function<D>,
    N2: Function<D>,

    N1::Value: CTrait<N2::Value, AXES>,
    N2::Value: Buffer<Field = FieldOf<N1::Value>>
{
    type Error = BinaryError<
        N1::Error,
        N2::Error,
        IncompatibleShapes<ShapeOf<N1::Value>, ShapeOf<N2::Value>>,
    >;
    type Value = <N1::Value as CTrait<N2::Value, AXES>>::Output;

    fn evaluate<DR: AsRef<D>>(&self, db: DR) -> Result<Self::Value, Self::Error> {
        let x = self.0.evaluate(db.as_ref()).map_err(BinaryError::Left)?;
        let y = self.1.evaluate(db).map_err(BinaryError::Right)?;

        x.contract(y).map_err(BinaryError::Output)
    }
}

impl<T, N1, N2, const AXES: usize> Differentiable<T> for Contract<AXES, N1, N2>
where
    T: Identifier,
    N1: Differentiable<T> + Clone,
    N2: Differentiable<T> + Clone,
{
    type Adjoint = Add<Contract<AXES, N1, N2::Adjoint>, Contract<AXES, N1::Adjoint, N2>>;

    fn adjoint(&self, target: T) -> Self::Adjoint {
        let j1 = self.0.adjoint(target);
        let j2 = self.1.adjoint(target);

        Add(Contract(self.0.clone(), j2), Contract(j1, self.1.clone()))
    }
}

impl<N1, N2, const AXES: usize> std::fmt::Debug for Contract<AXES, N1, N2>
where
    N1: std::fmt::Debug,
    N2: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple(&format!("Contract<{}>", AXES))
            .field(&self.0)
            .field(&self.1)
            .finish()
    }
}

/// Computes the product (contraction) of two tensor [Buffers](Buffer).
pub type TensorProduct<N1, N2> = Contract<0, N1, N2>;

impl<N1: std::fmt::Display, N2: std::fmt::Display> std::fmt::Display for TensorProduct<N1, N2> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({}) \u{2297} ({})", self.0, self.1)
    }
}

/// Computes the dot product (contraction) of two tensor [Buffers](Buffer).
///
/// # Examples
/// ```
/// # #[macro_use] extern crate aegir;
/// # use aegir::{Identifier, Differentiable, Dual, ops::TensorDot, ids::{X, Y}};
/// db!(DB { x: X, y: Y });
///
/// let f = TensorDot::new(X.into_var(), Y.into_var());
/// let db = DB {
///     x: [1.0, 2.0, 3.0],
///     y: [-1.0, 0.0, 2.0]
/// };
///
/// assert_eq!(f.evaluate_dual(X, &db).unwrap(), dual!(5.0, [-1.0, 0.0, 2.0]));
/// assert_eq!(f.evaluate_dual(Y, &db).unwrap(), dual!(5.0, [1.0, 2.0, 3.0]));
/// ```
pub type TensorDot<N1, N2> = Contract<1, N1, N2>;

impl<N1: std::fmt::Display, N2: std::fmt::Display> std::fmt::Display for TensorDot<N1, N2> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "\u{27E8}{}, {}\u{27E9}", self.0, self.1)
    }
}

/// Computes the double dot (contraction) of two tensor [Buffers](Buffer).
pub type TensorDoubleDot<N1, N2> = Contract<2, N1, N2>;

impl<N1: std::fmt::Display, N2: std::fmt::Display> std::fmt::Display for TensorDoubleDot<N1, N2> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({}) : ({})", self.0, self.1)
    }
}
