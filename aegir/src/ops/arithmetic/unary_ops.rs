use crate::{
    buffers::shapes::ShapeOf,
    buffers::{Buffer, FieldOf, Scalar, Spec},
    Contains,
    Database,
    Differentiable,
    Function,
    Identifier,
    Node,
};
use num_traits::real::Real;
use std::fmt;

impl_unary!(
    /// Computes the subtraction of one from a [Buffer].
    ///
    /// # Examples
    /// ```
    /// # #[macro_use] extern crate aegir;
    /// # use aegir::{Identifier, Differentiable, Dual, ops::SubOne, ids::X};
    /// db!(DB { x: X });
    ///
    /// let f = SubOne(X.into_var());
    ///
    /// assert_eq!(f.evaluate_dual(X, &DB { x: 1.0 }).unwrap(), dual!(0.0, 1.0));
    /// assert_eq!(f.evaluate_dual(X, &DB { x: 0.0 }).unwrap(), dual!(-1.0, 1.0));
    /// assert_eq!(f.evaluate_dual(X, &DB { x: -1.0 }).unwrap(), dual!(-2.0, 1.0));
    /// ```
    SubOne["({}) - 1"]: Scalar,
    |x| { x - num_traits::one() },
    |dx| { dx }
);

impl<N: Node> Node for SubOne<N> {}

impl<T, N> Differentiable<T> for SubOne<N>
where
    T: Identifier,
    N: Differentiable<T>,
{
    type Adjoint = N::Adjoint;

    fn adjoint(&self, target: T) -> Self::Adjoint { self.0.adjoint(target) }
}

/// Computes the subtraction of one from a [Buffer].
///
/// # Examples
/// ```
/// # #[macro_use] extern crate aegir;
/// # use aegir::{Identifier, Differentiable, Dual, ops::OneSub, ids::X};
/// db!(DB { x: X });
///
/// let f = OneSub(X.into_var());
///
/// assert_eq!(f.evaluate_dual(X, &DB { x: 1.0 }).unwrap(), dual!(0.0, -1.0));
/// assert_eq!(f.evaluate_dual(X, &DB { x: 0.0 }).unwrap(), dual!(1.0, -1.0));
/// assert_eq!(f.evaluate_dual(X, &DB { x: -1.0 }).unwrap(), dual!(2.0, -1.0));
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Contains)]
pub struct OneSub<N>(#[op] pub N);

impl<N: Node> Node for OneSub<N> {}

impl<D, N> crate::Function<D> for OneSub<N>
where
    D: Database,
    N: Function<D>,
{
    type Error = N::Error;
    type Value = N::Value;

    fn evaluate<DR: AsRef<D>>(&self, db: DR) -> Result<Self::Value, Self::Error> {
        self.0.evaluate(db).map(|buffer| {
            let o: FieldOf<N::Value> = num_traits::one();

            buffer.map(|x| o - x)
        })
    }
}

impl<X: Node + std::fmt::Display> std::fmt::Display for OneSub<X> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // if Stage::Instance(&self.0).is_zero() != TFU::True {
        write!(f, "1 - ({})", self.0)
        // } else {
        // write!(f, "1")
        // }
    }
}

impl<T, N> Differentiable<T> for OneSub<N>
where
    T: Identifier,
    N: Differentiable<T>,
{
    type Adjoint = Negate<N::Adjoint>;

    fn adjoint(&self, target: T) -> Self::Adjoint { Negate(self.0.adjoint(target)) }
}

impl_unary!(
    /// Adds one to a [Buffer].
    ///
    /// # Examples
    /// ```
    /// # #[macro_use] extern crate aegir;
    /// # use aegir::{Identifier, Differentiable, Dual, ops::AddOne, ids::X};
    /// db!(DB { x: X });
    ///
    /// let f = AddOne(X.into_var());
    ///
    /// assert_eq!(f.evaluate_dual(X, &DB { x: 1.0 }).unwrap(), dual!(2.0, 1.0));
    /// assert_eq!(f.evaluate_dual(X, &DB { x: 0.0 }).unwrap(), dual!(1.0, 1.0));
    /// assert_eq!(f.evaluate_dual(X, &DB { x: -1.0 }).unwrap(), dual!(0.0, 1.0));
    /// ```
    AddOne["({}) + 1"]: Scalar,
    |x| { x + num_traits::one() },
    |dx| { dx }
);

impl<N: Node> Node for AddOne<N> {}

impl<T, N> Differentiable<T> for AddOne<N>
where
    T: Identifier,
    N: Differentiable<T>,
{
    type Adjoint = N::Adjoint;

    fn adjoint(&self, target: T) -> Self::Adjoint { self.0.adjoint(target) }
}

/// Computes the square of a [Buffer].
///
/// # Examples
/// ```
/// # #[macro_use] extern crate aegir;
/// # use aegir::{Identifier, Differentiable, Dual, ops::Square, ids::X};
/// db!(DB { x: X });
///
/// let f = Square(X.into_var());
///
/// assert_eq!(f.evaluate_dual(X, &DB { x: -1.0 }).unwrap(), dual!(1.0, -2.0));
/// assert_eq!(f.evaluate_dual(X, &DB { x: 0.0 }).unwrap(), dual!(0.0, 0.0));
/// assert_eq!(f.evaluate_dual(X, &DB { x: 1.0 }).unwrap(), dual!(1.0, 2.0));
/// assert_eq!(f.evaluate_dual(X, &DB { x: 2.0 }).unwrap(), dual!(4.0, 4.0));
/// ```
#[derive(Copy, Clone, Debug, PartialEq, Contains)]
pub struct Square<N>(#[op] pub N);

impl<N: Node> Node for Square<N> {}

impl<F, D, N> Function<D> for Square<N>
where
    F: Scalar + num_traits::Pow<F, Output = F>,
    D: Database,
    N: Function<D, Value = F>,
{
    type Error = N::Error;
    type Value = F;

    fn evaluate<DR: AsRef<D>>(&self, db: DR) -> Result<Self::Value, Self::Error> {
        let two = num_traits::one::<F>() + num_traits::one();

        self.0.evaluate(db).map(|x| x.pow(two))
    }
}

impl<T, N> Differentiable<T> for Square<N>
where
    T: Identifier,
    N: Differentiable<T> + Clone,
{
    type Adjoint = crate::ops::Mul<Double<N>, N::Adjoint>;

    fn adjoint(&self, target: T) -> Self::Adjoint {
        crate::ops::Mul(Double(self.0.clone()), self.0.adjoint(target))
    }
}

impl<X: Node + fmt::Display> fmt::Display for Square<X> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // if Stage::Instance(&self.0).is_zero() != TFU::True {
        write!(f, "({})^2", self.0)
        // } else {
        // write!(f, "0")
        // }
    }
}

/// Computes the double of a [Buffer].
///
/// # Examples
/// ```
/// # #[macro_use] extern crate aegir;
/// # use aegir::{Identifier, Differentiable, Dual, ops::Double, ids::X};
/// db!(DB { x: X });
///
/// let f = Double(X.into_var());
///
/// assert_eq!(f.evaluate_dual(X, &DB { x: -1.0 }).unwrap(), dual!(-2.0, 2.0));
/// assert_eq!(f.evaluate_dual(X, &DB { x: 0.0 }).unwrap(), dual!(0.0, 2.0));
/// assert_eq!(f.evaluate_dual(X, &DB { x: 1.0 }).unwrap(), dual!(2.0, 2.0));
/// ```
#[derive(Copy, Clone, Debug, PartialEq, Contains)]
pub struct Double<N>(#[op] pub N);

impl<N: Node> Node for Double<N> {}

impl<D, N> Function<D> for Double<N>
where
    D: Database,
    N: Function<D>,
{
    type Error = N::Error;
    type Value = N::Value;

    fn evaluate<DR: AsRef<D>>(&self, db: DR) -> Result<Self::Value, Self::Error> {
        self.evaluate_spec(db).map(|state| state.unwrap())
    }

    fn evaluate_spec<DR: AsRef<D>>(&self, db: DR) -> Result<Spec<Self::Value>, Self::Error> {
        let two = num_traits::one::<FieldOf<N::Value>>() + num_traits::one();

        self.0
            .evaluate_spec(db)
            .map(|spec| spec.map(|x| two * x))
    }

    fn evaluate_shape<DR: AsRef<D>>(&self, db: DR) -> Result<ShapeOf<Self::Value>, Self::Error> {
        self.0.evaluate_shape(db)
    }
}

impl<T, N> Differentiable<T> for Double<N>
where
    T: Identifier,
    N: Differentiable<T>,
{
    type Adjoint = Double<N::Adjoint>;

    fn adjoint(&self, target: T) -> Self::Adjoint { Double(self.0.adjoint(target)) }
}

impl<X: Node + fmt::Display> fmt::Display for Double<X> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // if Stage::Instance(&self.0).is_zero() != TFU::True {
        write!(f, "2({})", self.0)
        // } else {
        // write!(f, "0")
        // }
    }
}

/// Compute the sum over elements in a [Buffer].
///
/// # Examples
/// ```
/// # #[macro_use] extern crate aegir;
/// # use aegir::{Identifier, Differentiable, Dual, ops::Sum, ids::X};
/// db!(DB { x: X });
///
/// let f = Sum(X.into_var());
///
/// assert_eq!(
///     f.evaluate_dual(X, &DB { x: [1.0, 2.0, 3.0] }).unwrap(),
///     dual!(6.0, [
///         [1.0, 0.0, 0.0],
///         [0.0, 1.0, 0.0],
///         [0.0, 0.0, 1.0]
///     ])
/// );
/// ```
#[derive(Copy, Clone, Debug, PartialEq, Contains)]
pub struct Sum<N>(#[op] pub N);

impl<N: Node> Node for Sum<N> {}

impl<D, N> Function<D> for Sum<N>
where
    D: Database,
    N: Function<D>,

    FieldOf<N::Value>: num_traits::Zero,
{
    type Error = N::Error;
    type Value = FieldOf<N::Value>;

    fn evaluate<DR: AsRef<D>>(&self, db: DR) -> Result<Self::Value, Self::Error> {
        self.0.evaluate(db).map(|buf| buf.sum())
    }
}

impl<T, N> Differentiable<T> for Sum<N>
where
    T: Identifier,
    N: Differentiable<T>,
{
    type Adjoint = N::Adjoint;

    fn adjoint(&self, target: T) -> Self::Adjoint { self.0.adjoint(target) }
}

impl<N: fmt::Display> fmt::Display for Sum<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "\u{03A3}_i ({})_i", self.0)
    }
}

impl_unary!(
    /// Computes the additive inverse of a [Buffer].
    ///
    /// # Examples
    /// ```
    /// # #[macro_use] extern crate aegir;
    /// # use aegir::{Identifier, Differentiable, Dual, ops::Negate, ids::X};
    /// db!(DB { x: X });
    ///
    /// let f = Negate(X.into_var());
    ///
    /// assert_eq!(f.evaluate_dual(X, &DB { x: 1.0 }).unwrap(), dual!(-1.0, -1.0));
    /// assert_eq!(f.evaluate_dual(X, &DB { x: -1.0 }).unwrap(), dual!(1.0, -1.0));
    /// ```
    Negate["-{}"]: num_traits::real::Real,
    |x| { -x },
    |dx| { -dx }
);

impl<N: Node> Node for Negate<N> {}

impl<T, N> Differentiable<T> for Negate<N>
where
    T: Identifier,
    N: Differentiable<T>,
{
    type Adjoint = Negate<N::Adjoint>;

    fn adjoint(&self, target: T) -> Self::Adjoint { Negate(self.0.adjoint(target)) }
}

fn dirac<F: Scalar + num_traits::Float>(x: F) -> F {
    match x {
        _ if (x == num_traits::zero()) => num_traits::Float::infinity(),
        _ => num_traits::zero(),
    }
}

impl_unary!(
    /// Computes the dirac delta about zero of a [Buffer].
    ///
    /// # Examples
    /// ```
    /// # #[macro_use] extern crate aegir;
    /// # use aegir::{Identifier, Function, Dual, ops::Dirac, ids::X};
    /// # use std::f64::INFINITY;
    /// db!(DB { x: X });
    ///
    /// let f = Dirac(X.into_var());
    ///
    /// assert_eq!(f.evaluate(&DB { x: -1.0 }).unwrap(), 0.0);
    /// assert_eq!(f.evaluate(&DB { x: 0.0 }).unwrap(), INFINITY);
    /// assert_eq!(f.evaluate(&DB { x: 1.0 }).unwrap(), 0.0);
    /// ```
    Dirac["\u{03B4}({})"]: num_traits::Float,
    dirac,
    |_| { unimplemented!() }
);

impl<N: Node> Node for Dirac<N> {}

/// Computes the sign of a [Buffer].
///
/// # Examples
/// ```
/// # #[macro_use] extern crate aegir;
/// # use aegir::{Identifier, Differentiable, Dual, ops::Sign, ids::X};
/// # use std::f64::INFINITY;
/// db!(DB { x: X });
///
/// let f = Sign(X.into_var());
///
/// assert_eq!(f.evaluate_dual(X, &DB { x: -1.0 }).unwrap(), dual!(-1.0, 0.0));
/// assert_eq!(f.evaluate_dual(X, &DB { x: 0.0 }).unwrap(), dual!(0.0, INFINITY));
/// assert_eq!(f.evaluate_dual(X, &DB { x: 1.0 }).unwrap(), dual!(1.0, 0.0));
/// ```
#[derive(Copy, Clone, Debug, PartialEq, Contains)]
pub struct Sign<N>(#[op] pub N);

impl<N: Node> Node for Sign<N> {}

impl<D: Database, N: Function<D>> Function<D> for Sign<N>
where
    FieldOf<N::Value>: num_traits::Float,
{
    type Error = N::Error;
    type Value = N::Value;

    fn evaluate<DR: AsRef<D>>(&self, db: DR) -> Result<Self::Value, Self::Error> {
        self.0.evaluate(db).map(|buffer| {
            buffer.map(|x| {
                if num_traits::Zero::is_zero(&x) {
                    x
                } else {
                    x.signum()
                }
            })
        })
    }
}

impl<T, N> Differentiable<T> for Sign<N>
where
    T: Identifier,
    N: Differentiable<T> + Clone,
{
    type Adjoint = crate::ops::Mul<Dirac<N>, N::Adjoint>;

    fn adjoint(&self, target: T) -> Self::Adjoint {
        Dirac(self.0.clone()).mul(self.0.adjoint(target))
    }
}

impl<X: fmt::Display> fmt::Display for Sign<X> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "sgn({})", self.0) }
}

/// Computes the absolute value of a [Buffer].
///
/// # Examples
/// ```
/// # #[macro_use] extern crate aegir;
/// # use aegir::{Identifier, Differentiable, Dual, ops::Abs, ids::X};
/// db!(DB { x: X });
///
/// let f = Abs(X.into_var());
///
/// assert_eq!(f.evaluate_dual(X, &DB { x: -1.0 }).unwrap(), dual!(1.0, -1.0));
/// assert_eq!(f.evaluate_dual(X, &DB { x: 0.0 }).unwrap(), dual!(0.0, 0.0));
/// assert_eq!(f.evaluate_dual(X, &DB { x: 1.0 }).unwrap(), dual!(1.0, 1.0));
/// ```
#[derive(Copy, Clone, Debug, PartialEq, Contains)]
pub struct Abs<N>(#[op] pub N);

impl<N: Node> Node for Abs<N> {}

impl<D: Database, N: Function<D>> Function<D> for Abs<N>
where
    FieldOf<N::Value>: num_traits::real::Real,
{
    type Error = N::Error;
    type Value = N::Value;

    fn evaluate<DR: AsRef<D>>(&self, db: DR) -> Result<Self::Value, Self::Error> {
        self.0.evaluate(db).map(|buffer| buffer.map(|x| x.abs()))
    }
}

impl<T, N> Differentiable<T> for Abs<N>
where
    T: Identifier,
    N: Differentiable<T> + Clone,
{
    type Adjoint = crate::ops::Mul<Sign<N>, N::Adjoint>;

    fn adjoint(&self, target: T) -> Self::Adjoint {
        Sign(self.0.clone()).mul(self.0.adjoint(target))
    }
}

impl<N: Node + fmt::Display> fmt::Display for Abs<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // if Stage::Instance(&self.0).is_zero() != TFU::True {
        write!(f, "|{}|", self.0)
        // } else {
        // write!(f, "0")
        // }
    }
}
