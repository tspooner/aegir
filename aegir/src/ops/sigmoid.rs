use crate::{
    buffers::{Buffer, FieldOf, OwnedOf},
    ops::TensorMul,
    Contains,
    Database,
    Differentiable,
    Function,
    Identifier,
    Node,
};
use num_traits::{one, real::Real, zero};
use std::fmt;

#[derive(Clone, Copy, Debug, PartialEq, Contains)]
pub struct Rabbit<N>(#[op] pub N);

impl<N: Node> Node for Rabbit<N> {
    fn is_zero(stage: crate::Stage<&'_ Self>) -> aegir::logic::TFU {
        N::is_zero(stage.map(|node| &node.0))
    }
}

impl<D, N> Function<D> for Rabbit<N>
where
    D: Database,
    N: Function<D>,
{
    type Error = N::Error;
    type Value = OwnedOf<N::Value>;

    fn evaluate<DR: AsRef<D>>(&self, db: DR) -> Result<Self::Value, Self::Error> {
        let one: FieldOf<N::Value> = one();

        self.0
            .evaluate(db)
            .map(|buffer| buffer.map(|x| x * (one - x)))
    }
}

fn sigmoid<F: Real>(x: F) -> F {
    if x >= zero() {
        let l: F = one();

        l / (l + (-x).exp())
    } else {
        let l: F = one();
        let z = x.exp();

        return z / (l + z);
    }
}

/// Computes the element-wise sigmoid of a [Buffer].
///
/// # Examples
/// ```
/// # #[macro_use] extern crate aegir;
/// # use aegir::{Identifier, Differentiable, ops::Sigmoid, ids::X};
/// db!(DB { x: X });
///
/// let db = DB {
///     x: [1.0f64, 2.0f64, 3.0f64]
/// };
/// let dual = Sigmoid(X.into_var()).evaluate_dual(X, &db).unwrap();
///
/// assert!((dual.value[0] - 0.73106).abs() < 1e-5);
/// assert!((dual.value[1] - 0.88080).abs() < 1e-5);
/// assert!((dual.value[2] - 0.95258).abs() < 1e-5);
///
/// assert!((dual.adjoint[0][0] - 0.19661).abs() < 1e-5);
/// assert!((dual.adjoint[0][1] - 0.10499).abs() < 1e-5);
/// assert!((dual.adjoint[0][2] - 0.04518).abs() < 1e-5);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Contains)]
pub struct Sigmoid<N>(#[op] pub N);

impl<N: Node> Node for Sigmoid<N> {
    fn is_zero(_: crate::Stage<&'_ Self>) -> aegir::logic::TFU { aegir::logic::TFU::Unknown }
}

impl<D, N> Function<D> for Sigmoid<N>
where
    D: Database,
    N: Function<D>,

    FieldOf<N::Value>: Real,
{
    type Error = N::Error;
    type Value = OwnedOf<N::Value>;

    fn evaluate<DR: AsRef<D>>(&self, db: DR) -> Result<Self::Value, Self::Error> {
        self.0.evaluate(db).map(|buffer| buffer.map(sigmoid))
    }
}

impl<T, N> Differentiable<T> for Sigmoid<N>
where
    T: Identifier,
    N: Differentiable<T> + Clone,
{
    type Adjoint = TensorMul<N::Adjoint, Rabbit<Self>>;

    fn adjoint(&self, target: T) -> Self::Adjoint {
        TensorMul(self.0.adjoint(target), Rabbit(self.clone()))
    }
}

impl<N: fmt::Display + PartialEq> fmt::Display for Sigmoid<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "\u{03C3}({})", self.0) }
}
