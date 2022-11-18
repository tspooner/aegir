use crate::{
    buffers::{
        shapes::{Concat, Shape},
        Buffer,
        Class,
        FieldOf,
        OwnedOf,
        Scalar,
        Scalars,
        Arrays,
        Vecs,
        Tuples,
    },
    Contains,
    Database,
    Differentiable,
    Function,
    Identifier,
    Node,
    Read,
};
use super::SourceError;

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
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Constant<B>(pub B);

impl<B> Node for Constant<B> {}

impl<T, B> Contains<T> for Constant<B>
where
    T: Identifier,
{
    fn contains(&self, _: T) -> bool { false }
}

impl<D, B> Function<D> for Constant<B>
where
    D: Database,
    B: Buffer,
{
    type Error = SourceError<()>;
    type Value = OwnedOf<B>;

    fn evaluate<DR: AsRef<D>>(&self, _: DR) -> Result<Self::Value, Self::Error> {
        Ok(self.0.to_owned())
    }
}

impl<T, B> Differentiable<T> for Constant<B>
where
    T: Identifier,
    B: Buffer,

    FieldOf<B>: num_traits::Zero,
{
    type Adjoint = ConstantAdjoint<Constant<OwnedOf<B>>, T>;

    fn adjoint(&self, ident: T) -> Self::Adjoint {
        ConstantAdjoint {
            node: Constant(self.0.to_owned()),
            target: ident,
        }
    }
}

impl<B: std::fmt::Display> std::fmt::Display for Constant<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { self.0.fmt(f) }
}

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
{
    type Error = crate::BinaryError<N::Error, SourceError<T>, crate::NoError>;
    type Value = <CA as Class<SA>>::Buffer<F>;

    fn evaluate<DR: AsRef<D>>(&self, db: DR) -> Result<Self::Value, Self::Error> {
        let shape_value = self
            .node
            .evaluate_shape(db.as_ref())
            .map_err(crate::BinaryError::Left)?;
        let shape_target = db.as_ref().read(self.target).map(|buf| buf.shape()).ok_or(
            crate::BinaryError::Right(SourceError::Undefined(self.target)),
        )?;
        let shape_adjoint = shape_value.concat(shape_target);

        Ok(CA::zeroes::<F>(shape_adjoint))
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
