use super::SourceError;
use crate::{
    buffers::{
        shapes::{Concat, Shape},
        Arrays,
        Buffer,
        Class,
        OwnedOf,
        ShapeOf,
        Scalar,
        Scalars,
        Tuples,
        Vecs,
    },
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
/// from the provided [Database] and returns the buffer assigned to `I`, and the
/// latter returns an an instance of [VariableAdjoint]. You should use this type
/// as the entry point for all "symbolic" entities in the constructed operator
/// tree.
///
/// # Examples
/// ```
/// # #[macro_use] extern crate aegir;
/// # use aegir::{Variable, Function, Differentiable, ids::X};
/// db!(DB { x: X });
///
/// let var = Variable(X);
/// let jac = var.adjoint(X);
///
/// assert_eq!(var.evaluate(DB { x: [1.0, 2.0] }).unwrap(), [1.0, 2.0]);
/// assert_eq!(jac.evaluate(DB { x: [1.0, 2.0] }).unwrap(), [
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

impl<D, I> Function<D> for Variable<I>
where
    D: Read<I>,
    I: Identifier,

    OwnedOf<D::Buffer>: Buffer<Shape = ShapeOf<D::Buffer>>,
{
    type Error = SourceError<I>;
    type Value = OwnedOf<D::Buffer>;

    fn evaluate<DR: AsRef<D>>(&self, db: DR) -> Result<Self::Value, Self::Error> {
        db.as_ref()
            .read(self.0)
            .map(|v| v.to_owned())
            .ok_or_else(|| SourceError::Undefined(self.0))
    }

    fn evaluate_shape<DR: AsRef<D>>(&self, db: DR) -> Result<ShapeOf<Self::Value>, Self::Error> {
        db.as_ref()
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

impl<I: PartialEq<T>, T> Node for VariableAdjoint<I, T> {
    fn is_zero(stage: aegir::Stage<&'_ Self>) -> aegir::logic::TFU {
        match stage {
            aegir::Stage::Evaluation(me) | aegir::Stage::Instance(me) => {
                if me.value == me.target { false.into() } else { true.into() }
            }
            _ => aegir::logic::TFU::Unknown,
        }
    }

    fn is_one(stage: aegir::Stage<&'_ Self>) -> aegir::logic::TFU {
        !Self::is_zero(stage)
    }
}

impl<I, T, A> Contains<A> for VariableAdjoint<I, T>
where
    I: Identifier + PartialEq<T> + PartialEq<A>,
    T: Identifier + PartialEq<A>,
    A: Identifier,
{
    fn contains(&self, ident: A) -> bool { self.value == ident || self.target == ident }
}

impl<I, T, D, F, SI, CI, ST, CT, SA, CA> Function<D> for VariableAdjoint<I, T>
where
    I: Identifier + PartialEq<T>,
    T: Identifier,

    D: Read<I> + Read<T>,
    F: Scalar,

    SI: Concat<SI> + Concat<ST, Shape = SA>,
    ST: Shape,
    SA: Shape,

    CI: Class<SI>,
    CT: Class<ST>,
    CA: Class<SA>,

    super::Prec: super::Precedence<CI, CT, Class = CA>,

    <D as Read<I>>::Buffer: Buffer<Class = CI, Shape = SI, Field = F>,
    <D as Read<T>>::Buffer: Buffer<Class = CT, Shape = ST, Field = F>,

    <CA as Class<SA>>::Buffer<F>: Buffer<Shape = SA>,
{
    type Error = crate::BinaryError<SourceError<I>, SourceError<T>, crate::NoError>;
    type Value = <CA as Class<SA>>::Buffer<F>;

    fn evaluate<DR: AsRef<D>>(&self, db: DR) -> Result<Self::Value, Self::Error> {
        let shape_value = db
            .as_ref()
            .read_shape(self.value)
            .ok_or(crate::BinaryError::Left(SourceError::Undefined(self.value)))?;
        let shape_target =
            db.as_ref()
                .read_shape(self.target)
                .ok_or(crate::BinaryError::Right(SourceError::Undefined(
                    self.target,
                )))?;
        let shape_adjoint = shape_value.concat(shape_target);

        Ok(if self.value == self.target {
            // In this case, we also know that shape_value == shape_target.
            // This further implies that shape_adjoint.split() is exactly equal
            // to (shape_value, shape_adjoint). We exploit this below:
            let one = num_traits::one();
            let ixs = shape_value
                .indices()
                .zip(shape_target.indices())
                .map(|ixs| <SI as Concat<ST>>::concat_indices(ixs.0, ixs.1));

            CA::build_subset(shape_adjoint, num_traits::zero(), ixs, |_| one)
        } else {
            <Self::Value as Buffer>::Class::full(shape_adjoint, num_traits::zero())
        })
    }

    fn evaluate_shape<DR: AsRef<D>>(&self, db: DR) -> Result<SA, Self::Error> {
        let shape_value = db
            .as_ref()
            .read_shape(self.value)
            .ok_or(crate::BinaryError::Left(SourceError::Undefined(self.value)))?;
        let shape_target =
            db.as_ref()
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

impl<I, T> std::fmt::Display for VariableAdjoint<I, T>
where
    I: std::fmt::Display + PartialEq<T>,
    T: std::fmt::Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.value == self.target {
            write!(f, "\u{2202}{}", self.value)
        } else {
            write!(f, "0")
        }
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

    #[derive(Database)]
    struct DB<A, B> {
        #[id(X)]
        pub x: A,

        #[id(Y)]
        pub y: B,
    }

    #[test]
    fn test_variable() {
        let var = X.into_var();

        assert_eq!(var.evaluate(&DB { x: 1.0, y: 0.0 }).unwrap(), 1.0);
        assert_eq!(
            var.evaluate(&DB {
                x: [-10.0, 5.0],
                y: 0.0
            })
            .unwrap(),
            [-10.0, 5.0]
        );
        assert_eq!(
            var.evaluate(&DB {
                x: (-1.0, 50.0),
                y: 0.0
            })
            .unwrap(),
            (-1.0, 50.0)
        );
        assert_eq!(
            var.evaluate(&DB {
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

        assert_eq!(g.evaluate(&DB { x: 1.0, y: 0.0 }).unwrap(), 0.0);
        // assert_eq!(g.evaluate(&DB { x: [-10.0, 5.0], y: 0.0 }).unwrap(),
        // [0.0; 2]); assert_eq!(g.evaluate(&DB { x: (-1.0, 50.0), y:
        // 0.0 }).unwrap(), (0.0, 0.0)); assert_eq!(g.evaluate(&DB { x:
        // vec![1.0, 2.0], y: 0.0 }).unwrap(), vec![0.0; 2]);
    }

    #[test]
    fn test_adjoint_one() {
        let g = X.into_var().adjoint(X);

        assert_eq!(g.evaluate(&DB { x: 1.0, y: 0.0 }).unwrap(), 1.0);
        // assert_eq!(g.evaluate(&DB { x: [-10.0, 5.0] }).unwrap(), [1.0; 2]);
        // assert_eq!(g.evaluate(&DB { x: (-1.0, 50.0) }).unwrap(), (1.0, 1.0));
        // assert_eq!(g.evaluate(&DB { x: vec![1.0, 2.0] }).unwrap(), vec![1.0;
        // 2]);
    }
}
