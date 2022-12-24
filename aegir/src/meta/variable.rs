use super::SourceError;
use crate::{
    buffers::{
        shapes::{Concat, Shape, ShapeOf},
        Buffer,
        Class,
        Scalar,
        Spec,
    },
    fmt::{Expr, PreWrap, ToExpr},
    Contains,
    Differentiable,
    Function,
    Identifier,
    Node,
    Read,
};

/// Source node for numerical variables.
///
/// This node implements both [Function] and [Differentiable]. The former reads
/// from the provided [Context] and returns the buffer assigned to `I`, and the
/// latter returns an an instance of [VariableAdjoint]. You should use this type
/// as the entry point for all "symbolic" entities in the constructed operator
/// tree.
///
/// # Examples
/// ```
/// # #[macro_use] extern crate aegir;
/// # use aegir::{Function, Differentiable, ids::X, meta::Variable};
/// ctx!(Ctx { x: X });
///
/// let var = Variable(X);
/// let jac = var.adjoint(X);
///
/// assert_eq!(var.evaluate(Ctx { x: [1.0, 2.0] }).unwrap(), [1.0, 2.0]);
/// assert_eq!(jac.evaluate(Ctx { x: [1.0, 2.0] }).unwrap(), [
///     [1.0, 0.0],
///     [0.0, 1.0]
/// ]);
/// ```
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Variable<I>(pub I);

impl<I> Node for Variable<I> {}

impl<T, I> Contains<T> for Variable<I>
where
    T: Identifier,
    I: Identifier + std::cmp::PartialEq<T>,
{
    fn contains(&self, ident: T) -> bool { self.0 == ident }
}

impl<C, I> Function<C> for Variable<I>
where
    C: Read<I>,
    I: Identifier,
{
    type Error = SourceError<I>;
    type Value = C::Buffer;

    fn evaluate<CR: AsRef<C>>(&self, ctx: CR) -> Result<Self::Value, Self::Error> {
        ctx.as_ref()
            .read(self.0)
            .ok_or_else(|| SourceError::Undefined(self.0))
    }

    fn evaluate_spec<CR: AsRef<C>>(&self, ctx: CR) -> Result<Spec<Self::Value>, Self::Error> {
        ctx.as_ref()
            .read_spec(self.0)
            .ok_or_else(|| SourceError::Undefined(self.0))
    }

    fn evaluate_shape<CR: AsRef<C>>(&self, ctx: CR) -> Result<ShapeOf<Self::Value>, Self::Error> {
        ctx.as_ref()
            .read_shape(self.0)
            .ok_or_else(|| SourceError::Undefined(self.0))
    }
}

impl<I, T> Differentiable<T> for Variable<I>
where
    I: Identifier + PartialEq<T>,
    T: Identifier,
{
    type Adjoint = VariableAdjoint<I, T>;

    fn adjoint(&self, target: T) -> Self::Adjoint {
        VariableAdjoint {
            value: self.0,
            target: target,
        }
    }
}

impl<I: Identifier> ToExpr for Variable<I> {
    fn to_expr(&self) -> Expr {
        Expr::Text(PreWrap {
            text: ToString::to_string(&self.0),
            needs_wrap: false,
        })
    }
}

impl<I: std::fmt::Display> std::fmt::Display for Variable<I> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.0, f)
    }
}

impl<I> From<I> for Variable<I> {
    fn from(selector: I) -> Variable<I> { Variable(selector) }
}

/// Source node for the adjoint of [variables](Variable).
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct VariableAdjoint<I, T> {
    /// The [Identifier] associated with the original [Variable].
    pub value: I,

    /// The [Identifier] associated with the adjoint target.
    pub target: T,
}

impl<I, T> Node for VariableAdjoint<I, T> {}

impl<I, T, A> Contains<A> for VariableAdjoint<I, T>
where
    I: Identifier + PartialEq<T> + PartialEq<A>,
    T: Identifier + PartialEq<A>,
    A: Identifier,
{
    fn contains(&self, ident: A) -> bool { self.value == ident || self.target == ident }
}

impl<I, T, C, F, SI, CI, ST, CT, SA, CA> Function<C> for VariableAdjoint<I, T>
where
    I: Identifier + PartialEq<T>,
    T: Identifier,

    C: Read<I> + Read<T>,
    F: Scalar,

    SI: Concat<SI> + Concat<ST, Shape = SA>,
    ST: Shape,
    SA: Shape,

    CI: Class<SI>,
    CT: Class<ST>,
    CA: Class<SA>,

    super::Prec: super::Precedence<CI, CT, Class = CA>,

    <C as Read<I>>::Buffer: Buffer<Class = CI, Shape = SI, Field = F>,
    <C as Read<T>>::Buffer: Buffer<Class = CT, Shape = ST, Field = F>,

    <CA as Class<SA>>::Buffer<F>: Buffer<Shape = SA>,
{
    type Error = crate::BinaryError<SourceError<I>, SourceError<T>, crate::NoError>;
    type Value = <CA as Class<SA>>::Buffer<F>;

    fn evaluate<CR: AsRef<C>>(&self, ctx: CR) -> Result<Self::Value, Self::Error> {
        self.evaluate_spec(ctx).map(|spec| spec.unwrap())
    }

    fn evaluate_spec<CR: AsRef<C>>(&self, ctx: CR) -> Result<Spec<Self::Value>, Self::Error> {
        let shape_value = ctx
            .as_ref()
            .read_shape(self.value)
            .ok_or(crate::BinaryError::Left(SourceError::Undefined(self.value)))?;
        let shape_target =
            ctx.as_ref()
                .read_shape(self.target)
                .ok_or(crate::BinaryError::Right(SourceError::Undefined(
                    self.target,
                )))?;
        let shape_adjoint = shape_value.concat(shape_target);

        Ok(if self.value == self.target {
            if SI::DIM + ST::DIM == 2 {
                Spec::eye(shape_adjoint)
            } else {
                // In this case, we also know that shape_value == shape_target.
                // This further implies that shape_adjoint.split() is exactly equal
                // to (shape_value, shape_adjoint). We exploit this below:
                let one = num_traits::one();
                let ixs = shape_value
                    .indices()
                    .zip(shape_target.indices())
                    .map(|ixs| <SI as Concat<ST>>::concat_indices(ixs.0, ixs.1));

                Spec::Raw(CA::build_subset(
                    shape_adjoint,
                    F::zero(),
                    ixs,
                    |_| one,
                ))
            }
        } else {
            Spec::zeroes(shape_adjoint)
        })
    }

    fn evaluate_shape<CR: AsRef<C>>(&self, ctx: CR) -> Result<SA, Self::Error> {
        let shape_value = ctx
            .as_ref()
            .read_shape(self.value)
            .ok_or(crate::BinaryError::Left(SourceError::Undefined(self.value)))?;
        let shape_target =
            ctx.as_ref()
                .read_shape(self.target)
                .ok_or(crate::BinaryError::Right(SourceError::Undefined(
                    self.target,
                )))?;

        Ok(shape_value.concat(shape_target))
    }
}

impl<I, T, A> Differentiable<A> for VariableAdjoint<I, T>
where
    I: PartialEq<T>,
    A: Identifier,

    Self: Clone,
{
    type Adjoint = super::ConstantAdjoint<Self, A>;

    fn adjoint(&self, ident: A) -> Self::Adjoint {
        super::ConstantAdjoint {
            node: self.clone(),
            target: ident,
        }
    }
}

impl<I, T> ToExpr for VariableAdjoint<I, T>
where
    I: Identifier + PartialEq<T>,
    T: Identifier,
{
    fn to_expr(&self) -> Expr {
        if self.value == self.target {
            Expr::One
        } else {
            Expr::Zero
        }
    }
}

impl<I, T> std::fmt::Display for VariableAdjoint<I, T>
where
    I: Identifier + PartialEq<T>,
    T: Identifier,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.to_expr().fmt(f)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aegir::{
        buffers::Buffer,
        ids::{X, Y},
        Function,
        Identifier,
    };

    #[derive(Context)]
    struct Ctx<A, B> {
        #[id(X)]
        pub x: A,

        #[id(Y)]
        pub y: B,
    }

    #[test]
    fn test_variable() {
        let var = X.into_var();

        assert_eq!(var.evaluate(&Ctx { x: 1.0, y: 0.0 }).unwrap(), 1.0);
        assert_eq!(
            var.evaluate(&Ctx {
                x: [-10.0, 5.0],
                y: 0.0
            })
            .unwrap(),
            [-10.0, 5.0]
        );
        assert_eq!(
            var.evaluate(&Ctx {
                x: (-1.0, 50.0),
                y: 0.0
            })
            .unwrap(),
            (-1.0, 50.0)
        );
        assert_eq!(
            var.evaluate(&Ctx {
                x: vec![1.0, 2.0],
                y: 0.0
            })
            .unwrap(),
            vec![1.0, 2.0]
        );
    }

    #[test]
    fn test_adjoint_zero() {
        let g = X.into_var().adjoint(Y);

        assert_eq!(g.evaluate(&Ctx { x: 1.0, y: 0.0 }).unwrap(), 0.0);
        // assert_eq!(g.evaluate(&Ctx { x: [-10.0, 5.0], y: 0.0 }).unwrap(),
        // [0.0; 2]); assert_eq!(g.evaluate(&Ctx { x: (-1.0, 50.0), y:
        // 0.0 }).unwrap(), (0.0, 0.0)); assert_eq!(g.evaluate(&Ctx { x:
        // vec![1.0, 2.0], y: 0.0 }).unwrap(), vec![0.0; 2]);
    }

    #[test]
    fn test_adjoint_one() {
        let g = X.into_var().adjoint(X);

        assert_eq!(g.evaluate(&Ctx { x: 1.0, y: 0.0 }).unwrap(), 1.0);
        // assert_eq!(g.evaluate(&Ctx { x: [-10.0, 5.0] }).unwrap(), [1.0; 2]);
        // assert_eq!(g.evaluate(&Ctx { x: (-1.0, 50.0) }).unwrap(), (1.0, 1.0));
        // assert_eq!(g.evaluate(&Ctx { x: vec![1.0, 2.0] }).unwrap(), vec![1.0;
        // 2]);
    }
}
