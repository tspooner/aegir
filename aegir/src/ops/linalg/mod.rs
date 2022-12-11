use crate::{
    buffers::{Buffer, Contract as CTrait, FieldOf, IncompatibleShapes, ShapeOf, shapes::Shape},
    ops::Add,
    BinaryError,
    Contains,
    Database,
    Differentiable,
    Function,
    Identifier,
    Node,
    State,
};

#[derive(Copy, Clone, PartialEq, Contains)]
pub struct Contract<const AXES: usize, L, R>(#[op] pub L, #[op] pub R);

impl<L, R, const AXES: usize> Contract<AXES, L, R> {
    pub fn new(left: L, right: R) -> Self { Contract(left, right) }
}

impl<L: Node, R: Node, const AXES: usize> Node for Contract<AXES, L, R> {}

impl<D, L, R, const AXES: usize> Function<D> for Contract<AXES, L, R>
where
    D: Database,
    L: Function<D>,
    R: Function<D>,

    L::Value: CTrait<R::Value, AXES>,
    R::Value: Buffer<Field = FieldOf<L::Value>>,
{
    type Error = BinaryError<
        L::Error,
        R::Error,
        IncompatibleShapes<ShapeOf<L::Value>, ShapeOf<R::Value>>,
    >;
    type Value = <L::Value as CTrait<R::Value, AXES>>::Output;

    fn evaluate<DR: AsRef<D>>(&self, db: DR) -> Result<Self::Value, Self::Error> {
        self.evaluate_state(db).map(|state| state.unwrap())
    }

    fn evaluate_shape<DR: AsRef<D>>(&self, db: DR) -> Result<ShapeOf<Self::Value>, Self::Error> {
        let x = self.0.evaluate_shape(&db).map_err(BinaryError::Left)?;
        let y = self.1.evaluate_shape(db).map_err(BinaryError::Right)?;

        <L::Value as CTrait<R::Value, AXES>>::contract_shape(x, y).map_err(BinaryError::Output)
    }

    fn evaluate_state<DR: AsRef<D>>(&self, db: DR) -> Result<State<Self::Value>, Self::Error> {
        let x = self.0.evaluate_state(&db).map_err(BinaryError::Left)?;
        let y = self.1.evaluate_state(db).map_err(BinaryError::Right)?;

        match (x, y) {
            // (State::Zero(sx), State::Zero(sy)) | (State::Zero(sx), State::One(sy)) | (State::One(sx), State::Zero(sy)) => {
                // <L::Value as CTrait<R::Value, AXES>>::contract_shape(sx, sy)
                    // .map(State::Zero)
                    // .map_err(BinaryError::Output)
            // },

            // (State::Zero(sx), State::Buffer(y)) => {
                // <L::Value as CTrait<R::Value, AXES>>::contract_shape(sx, y.shape())
                    // .map(State::Zero)
                    // .map_err(BinaryError::Output)
            // },

            // (State::Buffer(x), State::Zero(sy)) => {
                // <L::Value as CTrait<R::Value, AXES>>::contract_shape(x.shape(), sy)
                    // .map(State::Zero)
                    // .map_err(BinaryError::Output)
            // },

            (x, y) => {
                x.unwrap().contract(y.unwrap()).map(State::Buffer).map_err(BinaryError::Output)
            },
        }
    }
}

impl<T, L, R, const AXES: usize> Differentiable<T> for Contract<AXES, L, R>
where
    T: Identifier,
    L: Differentiable<T> + Clone,
    R: Differentiable<T> + Clone,
{
    type Adjoint = Add<Contract<AXES, L, R::Adjoint>, Contract<AXES, L::Adjoint, R>>;

    fn adjoint(&self, target: T) -> Self::Adjoint {
        let j1 = self.0.adjoint(target);
        let j2 = self.1.adjoint(target);

        Add(Contract(self.0.clone(), j2), Contract(j1, self.1.clone()))
    }
}

impl<L, R, const AXES: usize> std::fmt::Debug for Contract<AXES, L, R>
where
    L: std::fmt::Debug,
    R: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple(&format!("Contract<{}>", AXES))
            .field(&self.0)
            .field(&self.1)
            .finish()
    }
}

/// Computes the product (contraction) of two tensor [Buffers](Buffer).
pub type TensorProduct<L, R> = Contract<0, L, R>;

impl<L: std::fmt::Display, R: std::fmt::Display> std::fmt::Display for TensorProduct<L, R> {
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
pub type TensorDot<L, R> = Contract<1, L, R>;

impl<L: std::fmt::Display, R: std::fmt::Display> std::fmt::Display for TensorDot<L, R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "\u{27E8}{}, {}\u{27E9}", self.0, self.1)
    }
}

/// Computes the double dot (contraction) of two tensor [Buffers](Buffer).
pub type TensorDoubleDot<L, R> = Contract<2, L, R>;

impl<L: std::fmt::Display, R: std::fmt::Display> std::fmt::Display for TensorDoubleDot<L, R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({}) : ({})", self.0, self.1)
    }
}
