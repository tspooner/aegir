use crate::{
    buffers::{Compatible, Hadamard, OwnedOf, Scalar},
    ops::{HadOut, SafeXlnX},
    BinaryError,
    Contains,
    Database,
    Differentiable,
    Function,
    Identifier,
    Node,
};
use std::fmt;

/// Computes the power of a [Buffer] to a [Field].
///
/// # Examples
/// ## x^2
/// ```
/// # #[macro_use] extern crate aegir;
/// # use aegir::{Identifier, Differentiable, Dual, buffers::Buffer, ops::Power, ids::X};
/// db!(DB { x: X });
///
/// let f = Power(X.into_var(), 2.0f64.to_constant());
///
/// assert_eq!(f.evaluate_dual(X, &DB { x: -1.0 }).unwrap(), dual!(1.0, -2.0));
/// assert_eq!(f.evaluate_dual(X, &DB { x: 0.0 }).unwrap(), dual!(0.0, 0.0));
/// assert_eq!(f.evaluate_dual(X, &DB { x: 1.0 }).unwrap(), dual!(1.0, 2.0));
/// assert_eq!(f.evaluate_dual(X, &DB { x: 2.0 }).unwrap(), dual!(4.0, 4.0));
/// ```
///
/// ## x^y
/// ```
/// # #[macro_use] extern crate aegir;
/// # use aegir::{Identifier, Differentiable, Dual, buffers::Buffer, ops::Power, ids::{X, Y}};
/// # use aegir::{Function};
/// db!(DB { x: X, y: Y });
///
/// let f = Power(X.into_var(), Y.into_var());
///
/// assert!((
///     f.evaluate(&DB { x: 2.0, y: 1.5, }).unwrap() - 2.0f64.powf(1.5)
/// ) < 1e-5);
/// assert!((
///     f.evaluate_adjoint(X, &DB { x: 2.0, y: 1.5, }).unwrap() - 1.5 * 2.0f64.powf(0.5)
/// ) < 1e-5);
/// assert!((
///     f.evaluate_adjoint(Y, &DB { x: 2.0, y: 1.5, }).unwrap() - 2.0f64.powf(1.5) * 2.0f64.ln()
/// ) < 1e-5);
/// ```
///
/// # Derivation
/// We can derive the gradient computation as follows.
///
/// First, let
///      z(x) := f(x) ^ g(x).
/// Then, to get the derivative we first take logarithms,
///      ln z(x) = g(x) * ln f(x),
/// such that
///      z'(x) / z(x) = g'(x) * ln f(x) + f'(x) * g(x) / f(x).
/// It then follows that
///      z'(x) = z(x) * [g'(x) * ln f(x) + f'(x) * g(x) / f(x)]
/// as desired.
///
/// For simplicity, let
///      l(x) := z(x) * g'(x) * ln f(x),
/// and
///      r(x) := z(x) * f'(x) * g(x) / f(x).
/// Then
///      z'(x) = l(x) + r(x).
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Power<N, E>(pub N, pub E);

impl<N, E> Node for Power<N, E> {}

impl<T, N, E> Contains<T> for Power<N, E>
where
    T: Identifier,
    N: Contains<T>,
    E: Contains<T>,
{
    fn contains(&self, target: T) -> bool { self.0.contains(target) || self.1.contains(target) }
}

impl<F, D, N, E> Function<D> for Power<N, E>
where
    F: Scalar + num_traits::Pow<F, Output = F>,
    D: Database,

    N: Function<D, Value = F>,
    E: Function<D, Value = F>,
{
    type Error = BinaryError<N::Error, E::Error, crate::NoError>;
    type Value = OwnedOf<N::Value>;

    fn evaluate<DR: AsRef<D>>(&self, db: DR) -> Result<Self::Value, Self::Error> {
        let base = self.0.evaluate(db.as_ref()).map_err(BinaryError::Left)?;
        let exponent = self.1.evaluate(db).map_err(BinaryError::Right)?;

        Ok(base.map(|x| x.pow(exponent)))
    }
}

impl<T, N, E> Differentiable<T> for Power<N, E>
where
    T: Identifier,
    N: Differentiable<T> + Clone,
    E: Differentiable<T> + Clone,
{
    type Adjoint = Mul<
        Power<N, crate::ops::SubOne<E>>,
        Add<Mul<E, N::Adjoint>, Mul<SafeXlnX<N>, E::Adjoint>>,
    >;

    fn adjoint(&self, target: T) -> Self::Adjoint {
        let l = Power(self.0.clone(), crate::ops::SubOne(self.1.clone()));
        let r_l = Mul(self.1.clone(), self.0.adjoint(target));
        let r_r = Mul(SafeXlnX(self.0.clone()), self.1.adjoint(target));

        l.mul(r_l.add(r_r))
    }
}

impl<X: fmt::Display, E: fmt::Display> fmt::Display for Power<X, E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({})^({})", self.0, self.1)
    }
}

/// Computes the (elementwise) addition of two compatible [Buffer] types.
///
/// # Examples
/// ## x + y
/// ```
/// # #[macro_use] extern crate aegir;
/// # use aegir::{Identifier, Differentiable, Dual, ops::Add, ids::{X, Y}};
/// db!(DB { x: X, y: Y });
///
/// let f = Add(X.into_var(), Y.into_var());
///
/// assert_eq!(f.evaluate_dual(X, &DB { x: 1.0, y: 2.0, }).unwrap(), dual!(3.0, 1.0));
/// assert_eq!(f.evaluate_dual(Y, &DB { x: 1.0, y: 2.0, }).unwrap(), dual!(3.0, 1.0));
/// ```
///
/// ## x + y^2
/// ```
/// # #[macro_use] extern crate aegir;
/// # use aegir::{Identifier, Node, Differentiable, Dual, buffers::Buffer, ops::Add, ids::{X, Y}};
/// db!(DB { x: X, y: Y });
///
/// let f = Add(X.into_var(), Y.into_var().pow(2.0f64.to_constant()));
///
/// assert_eq!(f.evaluate_dual(X, &DB { x: 1.0, y: 2.0, }).unwrap(), dual!(5.0, 1.0));
/// assert_eq!(f.evaluate_dual(Y, &DB { x: 1.0, y: 2.0, }).unwrap(), dual!(5.0, 4.0));
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Node, Contains)]
pub struct Add<N1, N2>(#[op] pub N1, #[op] pub N2);

impl<D, N1, N2> Function<D> for Add<N1, N2>
where
    D: crate::Database,

    N1: crate::Function<D>,
    N2: crate::Function<D>,

    N1::Value: crate::buffers::Compatible<N2::Value>,
{
    type Error = BinaryError<
        N1::Error,
        N2::Error,
        crate::NoError, // IncompatibleBuffers<Pattern<N1::Value>, Pattern<N2::Value>>
    >;
    type Value = <N1::Value as crate::buffers::Hadamard<N2::Value>>::Output;

    fn evaluate<DR: AsRef<D>>(&self, db: DR) -> Result<Self::Value, Self::Error> {
        let x = self
            .0
            .evaluate(db.as_ref())
            .map_err(crate::BinaryError::Left)?;
        let y = self.1.evaluate(db).map_err(crate::BinaryError::Right)?;

        // x.hadamard(&y, $eval).map_err(BinaryError::Output)
        Ok(x.hadamard(&y, |xi, yi| xi + yi).unwrap())
    }
}

impl<T, N1, N2> Differentiable<T> for Add<N1, N2>
where
    T: Identifier,
    N1: Differentiable<T>,
    N2: Differentiable<T>,
{
    type Adjoint = Add<N1::Adjoint, N2::Adjoint>;

    fn adjoint(&self, target: T) -> Self::Adjoint {
        let x = self.0.adjoint(target);
        let y = self.1.adjoint(target);

        Add(x, y)
    }
}

impl<N1: std::fmt::Display, N2: std::fmt::Display> std::fmt::Display for Add<N1, N2> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({}) + ({})", self.0, self.1)
    }
}

/// Computes the (elementwise) subtraction of two compatible [Buffer] types.
///
/// # Examples
/// ## x - y
/// ```
/// # #[macro_use] extern crate aegir;
/// # use aegir::{Identifier, Differentiable, Dual, buffers::Buffer, ops::Sub, ids::{X, Y}};
/// db!(DB { x: X, y: Y });
///
/// let f = Sub(X.into_var(), Y.into_var());
///
/// assert_eq!(f.evaluate_dual(X, &DB { x: 1.0, y: 2.0, }).unwrap(), dual!(-1.0, 1.0));
/// assert_eq!(f.evaluate_dual(Y, &DB { x: 1.0, y: 2.0, }).unwrap(), dual!(-1.0, -1.0));
/// ```
///
/// ## x - y^2
/// ```
/// # #[macro_use] extern crate aegir;
/// # use aegir::{Identifier, Node, Differentiable, Dual, buffers::Buffer, ops::Sub, ids::{X, Y}};
/// db!(DB { x: X, y: Y });
///
/// let f = Sub(X.into_var(), Y.into_var().pow(2.0f64.to_constant()));
///
/// assert_eq!(f.evaluate_dual(X, &DB { x: 1.0, y: 2.0, }).unwrap(), dual!(-3.0, 1.0));
/// assert_eq!(f.evaluate_dual(Y, &DB { x: 1.0, y: 2.0, }).unwrap(), dual!(-3.0, -4.0));
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Node, Contains)]
pub struct Sub<N1, N2>(#[op] pub N1, #[op] pub N2);

impl<D, N1, N2> Function<D> for Sub<N1, N2>
where
    D: crate::Database,

    N1: crate::Function<D>,
    N2: crate::Function<D>,

    N1::Value: crate::buffers::Compatible<N2::Value>,
{
    type Error = BinaryError<
        N1::Error,
        N2::Error,
        crate::NoError, // IncompatibleBuffers<Pattern<N1::Value>, Pattern<N2::Value>>
    >;
    type Value = <N1::Value as crate::buffers::Hadamard<N2::Value>>::Output;

    fn evaluate<DR: AsRef<D>>(&self, db: DR) -> Result<Self::Value, Self::Error> {
        let x = self
            .0
            .evaluate(db.as_ref())
            .map_err(crate::BinaryError::Left)?;
        let y = self.1.evaluate(db).map_err(crate::BinaryError::Right)?;

        // x.hadamard(&y, $eval).map_err(BinaryError::Output)
        Ok(x.hadamard(&y, |xi, yi| xi - yi).unwrap())
    }
}

impl<T, N1, N2> Differentiable<T> for Sub<N1, N2>
where
    T: Identifier,
    N1: Differentiable<T>,
    N2: Differentiable<T>,
{
    type Adjoint = Sub<N1::Adjoint, N2::Adjoint>;

    fn adjoint(&self, target: T) -> Self::Adjoint {
        let x = self.0.adjoint(target);
        let y = self.1.adjoint(target);

        Sub(x, y)
    }
}

impl<N1: std::fmt::Display, N2: std::fmt::Display> std::fmt::Display for Sub<N1, N2> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({}) + ({})", self.0, self.1)
    }
}

/// Computes the (elementwise) product of two compatible [Buffer] types.
///
/// # Examples
/// ## x . y
/// ```
/// # #[macro_use] extern crate aegir;
/// # use aegir::{Identifier, Differentiable, Dual, buffers::Buffer, ops::Mul, ids::{X, Y}};
/// db!(DB { x: X, y: Y });
///
/// let f = Mul(X.into_var(), Y.into_var());
///
/// assert_eq!(f.evaluate_dual(X, &DB { x: 3.0, y: 2.0, }).unwrap(), dual!(6.0, 2.0));
/// assert_eq!(f.evaluate_dual(Y, &DB { x: 3.0, y: 2.0, }).unwrap(), dual!(6.0, 3.0));
/// ```
///
/// ## x . y^2
/// ```
/// # #[macro_use] extern crate aegir;
/// # use aegir::{Identifier, Node, Differentiable, Dual, buffers::Buffer, ops::Mul, ids::{X, Y}};
/// db!(DB { x: X, y: Y });
///
/// let f = Mul(X.into_var(), Y.into_var().pow(2.0f64.to_constant()));
///
/// assert_eq!(f.evaluate_dual(X, &DB { x: 3.0, y: 2.0, }).unwrap(), dual!(12.0, 4.0));
/// assert_eq!(f.evaluate_dual(Y, &DB { x: 3.0, y: 2.0, }).unwrap(), dual!(12.0, 12.0));
/// ```
#[derive(Copy, Clone, Debug, PartialEq, Node, Contains)]
pub struct Mul<N1, N2>(#[op] pub N1, #[op] pub N2);

impl<D, N1, N2> Function<D> for Mul<N1, N2>
where
    D: Database,
    N1: Function<D>,
    N2: Function<D>,

    N1::Value: Compatible<N2::Value>,
{
    type Error = BinaryError<N1::Error, N2::Error, crate::NoError>;
    type Value = HadOut<N1::Value, N2::Value>;

    fn evaluate<DR: AsRef<D>>(&self, db: DR) -> Result<Self::Value, Self::Error> {
        let x = self.0.evaluate(db.as_ref()).map_err(BinaryError::Left)?;
        let y = self.1.evaluate(db).map_err(BinaryError::Right)?;

        // x.hadamard(&y, |xi, yi| xi * yi).map_err(BinaryError::Output)
        Ok(x.hadamard(&y, |xi, yi| xi * yi).unwrap())
    }
}

impl<T, N1, N2> Differentiable<T> for Mul<N1, N2>
where
    T: Identifier,
    N1: Differentiable<T> + Clone,
    N2: Differentiable<T> + Clone,
{
    type Adjoint = Add<Mul<N1::Adjoint, N2>, Mul<N2::Adjoint, N1>>;

    fn adjoint(&self, target: T) -> Self::Adjoint {
        let gl = self.0.adjoint(target);
        let gr = self.1.adjoint(target);

        let ll = gl.mul(self.1.clone());
        let rr = gr.mul(self.0.clone());

        ll.add(rr)
    }
}

impl<N1: fmt::Display, N2: Node + fmt::Display> fmt::Display for Mul<N1, N2> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}) \u{2218} ({})", self.0, self.1)
    }
}

/// Computes the (elementwise) quotient of two compatible [Buffer] types.
#[derive(Copy, Clone, Debug, PartialEq, Node, Contains)]
pub struct Div<N1, N2>(#[op] pub N1, #[op] pub N2);

impl<D, N1, N2> Function<D> for Div<N1, N2>
where
    D: Database,
    N1: Function<D>,
    N2: Function<D>,

    N1::Value: Compatible<N2::Value>,
{
    type Error = BinaryError<N1::Error, N2::Error, crate::NoError>;
    type Value = HadOut<N1::Value, N2::Value>;

    fn evaluate<DR: AsRef<D>>(&self, db: DR) -> Result<Self::Value, Self::Error> {
        let x = self.0.evaluate(db.as_ref()).map_err(BinaryError::Left)?;
        let y = self.1.evaluate(db).map_err(BinaryError::Right)?;

        // x.hadamard(&y, |xi, yi| xi * yi).map_err(BinaryError::Output)
        Ok(x.hadamard(&y, |xi, yi| xi / yi).unwrap())
    }
}

// impl<F, D, T, N1, N2> Differentiable<D, T> for Div<N1, N2>
// where
// F: Scalar,
// D: Database,
// T: Identifier,
// N1: Differentiable<D, T>,
// N2: Differentiable<D, T>,

// N1::Value: Buffer<Field = F> + Hadamard<N2::Value> + Hadamard<N2::Adjoint,
// Output = N1::Value>, N2::Value: Buffer<Field = F> + Hadamard<N1::Adjoint,
// Output = N2::Value>,

// HadOut<N1::Value, N2::Value>: Hadamard<N2::Value>,
// {
// type Adjoint = HadOut<HadOut<N1::Value, N2::Value>, N2::Value>;

// fn grad(&self, db: &D, target: T) -> Result<Self::Adjoint, Self::Error> {
// let x = self.0.dual(db, target).map_err(BinaryError::Left)?;
// let y = self.1.dual(db, target).map_err(BinaryError::Right)?;

// let t1 = y.value.hadamard(&x.adjoint, |xi, yi| xi * yi)
// .unwrap();
// // .map_err(BinaryError::Output)?;
// let t2 = x.value.hadamard(&y.adjoint, |xi, yi| xi * yi)
// .unwrap();
// // .map_err(BinaryError::Output)?;

// let numerator = t2.hadamard(&t1, |yi, xi| xi - yi).unwrap();

// // t1.hadamard(&t2, |t1i, t2i| t1i + t2i).map_err(BinaryError::Output)
// Ok(numerator.hadamard(&y.value, |n, d| n / d).unwrap())
// }
// }

impl<T, N1, N2> Differentiable<T> for Div<N1, N2>
where
    T: Identifier,
    N1: Differentiable<T> + Clone,
    N2: Differentiable<T> + Clone,
{
    type Adjoint = Div<Sub<Mul<N2, N1::Adjoint>, Mul<N1, N2::Adjoint>>, crate::ops::Square<N2>>;

    fn adjoint(&self, target: T) -> Self::Adjoint {
        let l = self.0.adjoint(target);
        let r = self.1.adjoint(target);

        let numerator = self.1.clone().mul(l).sub(self.0.clone().mul(r));

        numerator.div(self.1.clone().squared())
    }
}

impl<N1: fmt::Display, N2: Node + fmt::Display> fmt::Display for Div<N1, N2> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}) / ({})", self.0, self.1)
    }
}
