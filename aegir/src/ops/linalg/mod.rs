use crate::{
    buffers::shapes::ShapeOf,
    buffers::{Buffer, Contract as CTrait, FieldOf, IncompatibleShapes, Spec},
    fmt::{ToExpr, Expr, PreWrap},
    ops::Add,
    BinaryError,
    Contains,
    Context,
    Differentiable,
    Function,
    Identifier,
    Node,
};

#[derive(Copy, Clone, PartialEq, Contains)]
pub struct Contract<const AXES: usize, L, R>(#[op] pub L, #[op] pub R);

impl<L, R, const AXES: usize> Contract<AXES, L, R> {
    pub fn new(left: L, right: R) -> Self { Contract(left, right) }
}

impl<L: Node, R: Node, const AXES: usize> Node for Contract<AXES, L, R> {}

impl<C, L, R, const AXES: usize> Function<C> for Contract<AXES, L, R>
where
    C: Context,
    L: Function<C>,
    R: Function<C>,

    L::Value: CTrait<R::Value, AXES>,
    R::Value: Buffer<Field = FieldOf<L::Value>>,
{
    type Error =
        BinaryError<L::Error, R::Error, IncompatibleShapes<ShapeOf<L::Value>, ShapeOf<R::Value>>>;
    type Value = <L::Value as CTrait<R::Value, AXES>>::Output;

    fn evaluate<CR: AsRef<C>>(&self, ctx: CR) -> Result<Self::Value, Self::Error> {
        self.evaluate_spec(ctx).map(|state| state.unwrap())
    }

    fn evaluate_spec<CR: AsRef<C>>(&self, ctx: CR) -> Result<Spec<Self::Value>, Self::Error> {
        let x = self.0.evaluate_spec(&ctx).map_err(BinaryError::Left)?;
        let y = self.1.evaluate_spec(ctx).map_err(BinaryError::Right)?;

        <L::Value as CTrait<R::Value, AXES>>::contract_spec(x, y).map_err(BinaryError::Output)
    }

    fn evaluate_shape<CR: AsRef<C>>(&self, ctx: CR) -> Result<ShapeOf<Self::Value>, Self::Error> {
        let x = self.0.evaluate_shape(&ctx).map_err(BinaryError::Left)?;
        let y = self.1.evaluate_shape(ctx).map_err(BinaryError::Right)?;

        <L::Value as CTrait<R::Value, AXES>>::contract_shape(x, y).map_err(BinaryError::Output)
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

/// Operator alias that applies the tensor product between two buffers.
pub type TensorProduct<L, R> = Contract<0, L, R>;

impl<L: ToExpr, R: ToExpr> ToExpr for TensorProduct<L, R> {
    fn to_expr(&self) -> Expr {
        use Expr::*;

        match (self.0.to_expr(), self.1.to_expr()) {
            (_, Zero) | (Zero, _) => Zero,
            (One, One) => One,

            (l, One) => l,
            (One, r) => r,

            (Text(l), Text(r)) => Text(PreWrap {
                text: format!("{} \u{2297} {}", l.to_safe_string('(', ')'), r.to_safe_string('(', ')')),
                needs_wrap: true,
            })
        }
    }
}

impl<L: ToExpr, R: ToExpr> std::fmt::Display for TensorProduct<L, R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.to_expr().fmt(f)
    }
}

/// Operator alias that applies the tensor dot product between two buffers.
///
/// # Examples
/// ```
/// # #[macro_use] extern crate aegir;
/// # use aegir::{Identifier, Differentiable, Dual, ops::TensorDot, ids::{X, Y}};
/// let f = TensorDot::new(X.into_var(), Y.into_var());
/// let ctx = ctx!{
///     X = [1.0, 2.0, 3.0],
///     Y = [-1.0, 0.0, 2.0]
/// };
///
/// assert_eq!(f.evaluate_dual(X, &ctx).unwrap(), dual!(5.0, [-1.0, 0.0, 2.0]));
/// assert_eq!(f.evaluate_dual(Y, &ctx).unwrap(), dual!(5.0, [1.0, 2.0, 3.0]));
/// ```
pub type TensorDot<L, R> = Contract<1, L, R>;

impl<L: ToExpr, R: ToExpr> ToExpr for TensorDot<L, R> {
    fn to_expr(&self) -> Expr {
        use Expr::*;

        match (self.0.to_expr(), self.1.to_expr()) {
            (_, Zero) | (Zero, _) => Zero,
            (One, One) => One,

            (l, One) => l,
            (One, r) => r,

            (Text(l), Text(r)) => Text(PreWrap {
                text: format!("\u{27E8}{}, {}\u{27E9}", l.to_safe_string('(', ')'), r.to_safe_string('(', ')')),
                needs_wrap: false,
            })
        }
    }
}

impl<L: ToExpr, R: ToExpr> std::fmt::Display for TensorDot<L, R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.to_expr().fmt(f)
    }
}

/// Operator alias that applies the tensor double dot product between two buffers.
pub type TensorDoubleDot<L, R> = Contract<2, L, R>;

impl<L: ToExpr, R: ToExpr> ToExpr for TensorDoubleDot<L, R> {
    fn to_expr(&self) -> Expr {
        use Expr::*;

        match (self.0.to_expr(), self.1.to_expr()) {
            (_, Zero) | (Zero, _) => Zero,
            (One, One) => One,

            (l, One) => l,
            (One, r) => r,

            (Text(l), Text(r)) => Text(PreWrap {
                text: format!("{} : {}", l.to_safe_string('(', ')'), r.to_safe_string('(', ')')),
                needs_wrap: true,
            })
        }
    }
}

impl<L: std::fmt::Display, R: std::fmt::Display> std::fmt::Display for TensorDoubleDot<L, R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({}) : ({})", self.0, self.1)
    }
}
