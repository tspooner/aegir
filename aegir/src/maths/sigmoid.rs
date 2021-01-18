use crate::{
    buffer::{Buffer, FieldOf, OwnedOf},
    dual::Dual,
    maths::MulOut,
    Contains,
    Database,
    Differentiable,
    DualOf,
    Function,
    Identifier,
};
use num_traits::{one, real::Real, zero};
use std::fmt;

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
/// # #[macro_use] extern crate ndarray;
/// # use aegir::{SimpleDatabase, Identifier, Differentiable, maths::Sigmoid};
/// ids!(X::x);
///
/// let db = SimpleDatabase::new(X, array![1.0, 2.0, 3.0]);
/// let dual = Sigmoid(X.to_var()).dual(&db, X).unwrap();
///
/// assert!((dual.value[0] - 0.73106).abs() < 1e-5);
/// assert!((dual.value[1] - 0.88080).abs() < 1e-5);
/// assert!((dual.value[2] - 0.95258).abs() < 1e-5);
///
/// assert!((dual.adjoint[0] - 0.19661).abs() < 1e-5);
/// assert!((dual.adjoint[1] - 0.10499).abs() < 1e-5);
/// assert!((dual.adjoint[2] - 0.04518).abs() < 1e-5);
/// ```
#[derive(Clone, Copy, Debug, Node, Contains)]
pub struct Sigmoid<N>(#[op] pub N);

impl<D, N> Function<D> for Sigmoid<N>
where
    D: Database,
    N: Function<D>,

    FieldOf<N::Codomain>: Real,
{
    type Codomain = OwnedOf<N::Codomain>;
    type Error = N::Error;

    fn evaluate(&self, db: &D) -> Result<Self::Codomain, Self::Error> {
        self.0.evaluate(db).map(|buffer| buffer.map(sigmoid))
    }
}

impl<D, T, N> Differentiable<D, T> for Sigmoid<N>
where
    D: Database,
    T: Identifier,
    N: Differentiable<D, T>,

    FieldOf<N::Codomain>: Real,
    FieldOf<N::Jacobian>: Real,

    N::Jacobian: std::ops::Mul<OwnedOf<N::Codomain>>,

    MulOut<N::Jacobian, OwnedOf<N::Codomain>>: Buffer<Field = FieldOf<N::Codomain>>,
{
    type Jacobian = MulOut<N::Jacobian, OwnedOf<N::Codomain>>;

    fn dual(&self, db: &D, target: T) -> Result<DualOf<Self, D, T>, Self::Error> {
        let o: FieldOf<N::Codomain> = one();

        self.0.dual(db, target).map(|dual| Dual {
            value: dual.value.to_owned().map(sigmoid),
            adjoint: dual.adjoint
                * dual.value.map(|x| {
                    let s = sigmoid(x);

                    s * (o - s)
                }),
        })
    }
}

impl<N: fmt::Display> fmt::Display for Sigmoid<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "\u{03C3}{}", self.0) }
}
