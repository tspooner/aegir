use super::SourceError;
use crate::{
    buffers::{
        shapes::{Concat, Shape, ShapeOf, Shaped},
        Buffer,
        Class,
        FieldOf,
        IntoSpec,
        Scalar,
        Spec,
    },
    Contains,
    Database,
    Differentiable,
    Function,
    Identifier,
    Node,
    Read,
};

/// Source node for numerical constants.
///
/// This node implements both [Function] and [Differentiable]. The former
/// simply returns the wrapped value, and the latter returns an instance of
/// [ConstantAdjoint] that handles (empty) buffer shaping.
///
/// # Examples
/// ```
/// # #[macro_use] extern crate aegir;
/// # use aegir::{Constant, Function, Differentiable, ids::X};
/// db!(DB { x: X });
///
/// let cns = Constant([10.0, 10.0]);
/// let jac = cns.adjoint(X);
///
/// assert_eq!(cns.evaluate(DB { x: [1.0, 2.0] }).unwrap(), [10.0, 10.0]);
/// assert_eq!(jac.evaluate(DB { x: [1.0, 2.0] }).unwrap(), [
///     [0.0, 0.0],
///     [0.0, 0.0]
/// ]);
/// ```
#[derive(Clone)]
pub struct Constant<S: Shaped + IntoSpec>(pub S);

impl<S: Shaped + IntoSpec> Node for Constant<S> {}

impl<T, S> Contains<T> for Constant<S>
where
    T: Identifier,
    S: Shaped + IntoSpec,
{
    fn contains(&self, _: T) -> bool { false }
}

impl<D, S> Function<D> for Constant<S>
where
    D: Database,
    S: Clone + Shaped + IntoSpec,
    S::Buffer: Shaped<Shape = S::Shape>,
{
    type Error = SourceError<()>;
    type Value = S::Buffer;

    fn evaluate<DR: AsRef<D>>(&self, db: DR) -> Result<S::Buffer, Self::Error> {
        self.evaluate_spec(db).map(|spec| spec.unwrap())
    }

    fn evaluate_spec<DR: AsRef<D>>(&self, _: DR) -> Result<Spec<S::Buffer>, Self::Error> {
        Ok(self.0.clone().into_spec())
    }

    fn evaluate_shape<DR: AsRef<D>>(&self, _: DR) -> Result<S::Shape, Self::Error> {
        Ok(self.0.shape())
    }
}

impl<T, B> Differentiable<T> for Constant<B>
where
    T: Identifier,
    B: Buffer,

    FieldOf<B>: num_traits::Zero,
{
    type Adjoint = ConstantAdjoint<Constant<B>, T>;

    fn adjoint(&self, ident: T) -> Self::Adjoint {
        ConstantAdjoint {
            node: self.clone(),
            target: ident,
        }
    }
}

impl<B> std::fmt::Debug for Constant<B>
where
    B: Buffer + std::fmt::Debug,
    B::Field: std::fmt::Debug,
    B::Shape: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { self.0.fmt(f) }
}

// impl<B: std::fmt::Display> std::fmt::Display for Constant<B> {
// fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
// self.0.fmt(f) } }

/// Source node for the adjoint of [constants](Constant).
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct ConstantAdjoint<N, T> {
    /// The original source [Node].
    pub node: N,

    /// The [Identifier] associated with the adjoint target.
    pub target: T,
}

impl<N: Node, T> Node for ConstantAdjoint<N, T> {}

impl<N, T, A> Contains<A> for ConstantAdjoint<N, T>
where
    N: Contains<A>,
    T: Identifier + PartialEq<A>,
    A: Identifier,
{
    fn contains(&self, ident: A) -> bool { self.node.contains(ident) || self.target == ident }
}

impl<N, T, D, F, SN, CN, ST, CT, SA, CA> Function<D> for ConstantAdjoint<N, T>
where
    F: Scalar,

    SN: Concat<ST, Shape = SA>,
    ST: Shape,
    SA: Shape,

    CN: Class<SN>,
    CT: Class<ST>,
    CA: Class<SA>,

    super::Prec: super::Precedence<CN, CT, Class = CA>,

    N: Function<D>,
    N::Value: Buffer<Field = F, Shape = SN, Class = CN>,

    T: Identifier,

    D: Read<T>,
    D::Buffer: Buffer<Field = F, Shape = ST, Class = CT>,

    <CA as Class<SA>>::Buffer<F>: Buffer<Shape = SA>,
{
    type Error = crate::BinaryError<N::Error, SourceError<T>, crate::NoError>;
    type Value = <CA as Class<SA>>::Buffer<F>;

    fn evaluate<DR: AsRef<D>>(&self, db: DR) -> Result<Self::Value, Self::Error> {
        self.evaluate_spec(db).map(|lifted| lifted.unwrap())
    }

    fn evaluate_spec<DR: AsRef<D>>(&self, db: DR) -> Result<Spec<Self::Value>, Self::Error> {
        self.evaluate_shape(db)
            .map(|shape| Spec::Full(shape, F::zero()))
    }

    fn evaluate_shape<DR: AsRef<D>>(&self, db: DR) -> Result<SA, Self::Error> {
        let shape_value = self
            .node
            .evaluate_shape(db.as_ref())
            .map_err(crate::BinaryError::Left)?;
        let shape_target = db.as_ref().read(self.target).map(|buf| buf.shape()).ok_or(
            crate::BinaryError::Right(SourceError::Undefined(self.target)),
        )?;

        Ok(shape_value.concat(shape_target))
    }
}

impl<N, T> std::fmt::Display for ConstantAdjoint<N, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { write!(f, "0") }
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
    fn test_constant() {
        let c = 2.0f64.into_constant();

        assert_eq!(c.evaluate(&DB { x: 1.0, y: 0.0 }).unwrap(), 2.0);
        assert_eq!(
            c.evaluate(&DB {
                x: [-10.0, 5.0],
                y: 0.0
            })
            .unwrap(),
            2.0
        );
        assert_eq!(
            c.evaluate(&DB {
                x: (-1.0, 50.0),
                y: 0.0
            })
            .unwrap(),
            2.0
        );
        assert_eq!(
            c.evaluate(&DB {
                x: vec![1.0, 2.0],
                y: 0.0
            })
            .unwrap(),
            2.0
        );
    }
}
