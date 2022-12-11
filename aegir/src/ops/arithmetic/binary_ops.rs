use crate::{
    buffers::{Buffer, OwnedOf, ShapeOf, Scalar, ZipMap, IncompatibleShapes, shapes::{self, Shape, Zip}, Class},
    logic::TFU,
    ops::SafeXlnX,
    BinaryError,
    Contains,
    Database,
    Differentiable,
    Function,
    Identifier,
    Node,
    Stage,
};
use std::fmt;

type Error<D, L, R> = BinaryError<
    <L as Function<D>>::Error,
    <R as Function<D>>::Error,
    IncompatibleShapes<ShapeOf<<L as Function<D>>::Value>, ShapeOf<<R as Function<D>>::Value>>
>;

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
#[derive(Copy, Clone, Debug, PartialEq, Contains)]
pub struct Power<N, E>(#[op] pub N, #[op] pub E);

impl<N: Node, E: Node> Node for Power<N, E> {
    fn is_one(stage: Stage<&'_ Self>) -> TFU { stage.map(|node| &node.0).is_one() }
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

    fn evaluate_shape<DR: AsRef<D>>(&self, _: DR) -> Result<shapes::S0, Self::Error> {
        Ok(shapes::S0)
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
#[derive(Clone, Copy, Debug, PartialEq, Contains)]
pub struct Add<L, R>(#[op] pub L, #[op] pub R);

impl<L: Node, R: Node> Node for Add<L, R> {
    fn is_zero(stage: Stage<&'_ Self>) -> TFU {
        (stage.map(|node| &node.0).is_zero() & stage.map(|node| &node.1).is_zero())
            .true_or(TFU::Unknown)
    }

    fn is_one(stage: Stage<&'_ Self>) -> TFU {
        let a = (stage.map(|node| &node.0).is_one() & stage.map(|node| &node.1).is_zero())
            .true_or(TFU::Unknown);
        let b = (stage.map(|node| &node.0).is_zero() & stage.map(|node| &node.1).is_one())
            .true_or(TFU::Unknown);

        a | b
    }
}

impl<D, F, L, LV, R, RV, OV> Function<D> for Add<L, R>
where
    D: Database,
    F: Scalar,

    L: Function<D, Value = LV>,
    LV: Buffer<Field = F> + ZipMap<RV, Output<F> = OV>,

    R: Function<D, Value = RV>,
    RV: Buffer<Field = F> + ZipMap<LV, Output<F> = OV>,

    OV: Buffer<Field = F>,

    LV::Shape: Zip<RV::Shape, Shape = OV::Shape>,
{
    type Error = Error<D, L, R>;
    type Value = OV;

    fn evaluate<DR: AsRef<D>>(&self, db: DR) -> Result<Self::Value, Self::Error> {
        use crate::Stage::Evaluation;

        // First establish if either term is zero...
        if Evaluation(&self.0).is_zero() == TFU::True {
            return short_circuit!(@TakeRight db[self.0, self.1]);
        }

        // If not, then maybe the second term is...
        if Evaluation(&self.1).is_zero() == TFU::True {
            return short_circuit!(@TakeLeft db[self.0, self.1]);
        }

        // Otherwise, we do the actual addition...
        let x = self
            .0
            .evaluate(db.as_ref())
            .map_err(crate::BinaryError::Left)?;
        let y = self.1.evaluate(db).map_err(crate::BinaryError::Right)?;

        Ok(x.zip_map(y, |xi, yi| xi + yi).unwrap())
    }

    fn evaluate_shape<DR: AsRef<D>>(&self, db: DR) -> Result<ShapeOf<Self::Value>, Self::Error> {
        let l_shape = self.0.evaluate_shape(&db).map_err(BinaryError::Left)?;
        let r_shape = self.1.evaluate_shape(db).map_err(BinaryError::Right)?;

        l_shape.zip(r_shape).map_err(BinaryError::Output)
    }
}

impl<T, L, R> Differentiable<T> for Add<L, R>
where
    T: Identifier,
    L: Differentiable<T>,
    R: Differentiable<T>,
{
    type Adjoint = Add<L::Adjoint, R::Adjoint>;

    fn adjoint(&self, target: T) -> Self::Adjoint {
        let x = self.0.adjoint(target);
        let y = self.1.adjoint(target);

        Add(x, y)
    }
}

impl<L: Node + std::fmt::Display, R: Node + std::fmt::Display> std::fmt::Display
    for Add<L, R>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use crate::Stage::Evaluation;

        if Evaluation(&self.0).is_zero() == TFU::True {
            write!(f, "{}", self.1)
        } else if Evaluation(&self.1).is_zero() == TFU::True {
            write!(f, "{}", self.0)
        } else {
            write!(f, "({}) + ({})", self.0, self.1)
        }
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
#[derive(Clone, Copy, Debug, PartialEq, Contains)]
pub struct Sub<L, R>(#[op] pub L, #[op] pub R);

impl<L: Node, R: Node> Node for Sub<L, R> {
    fn is_zero(stage: Stage<&'_ Self>) -> TFU {
        let a = (stage.map(|node| &node.0).is_one() & stage.map(|node| &node.1).is_one())
            .true_or(TFU::Unknown);
        let b = (stage.map(|node| &node.0).is_zero() & stage.map(|node| &node.1).is_zero())
            .true_or(TFU::Unknown);

        a | b
    }

    fn is_one(stage: Stage<&'_ Self>) -> TFU {
        (stage.map(|node| &node.0).is_one() & stage.map(|node| &node.1).is_zero())
            .true_or(TFU::Unknown)
    }
}

impl<D, F, L, LV, R, RV, OV> Function<D> for Sub<L, R>
where
    D: Database,
    F: Scalar,

    L: Function<D, Value = LV>,
    LV: Buffer<Field = F> + ZipMap<RV, Output<F> = OV>,

    R: Function<D, Value = RV>,
    RV: Buffer<Field = F> + ZipMap<LV, Output<F> = OV>,

    OV: Buffer<Field = F>,

    LV::Shape: Zip<RV::Shape, Shape = OV::Shape>,
{
    type Error = Error<D, L, R>;
    type Value = OV;

    fn evaluate<DR: AsRef<D>>(&self, db: DR) -> Result<Self::Value, Self::Error> {
        let x = self
            .0
            .evaluate(db.as_ref())
            .map_err(crate::BinaryError::Left)?;
        let y = self.1.evaluate(db).map_err(crate::BinaryError::Right)?;

        x.zip_map(y, |xi, yi| xi - yi).map_err(BinaryError::Output)
    }

    fn evaluate_shape<DR: AsRef<D>>(&self, db: DR) -> Result<ShapeOf<Self::Value>, Self::Error> {
        let l_shape = self.0.evaluate_shape(&db).map_err(BinaryError::Left)?;
        let r_shape = self.1.evaluate_shape(db).map_err(BinaryError::Right)?;

        l_shape.zip(r_shape).map_err(BinaryError::Output)
    }
}

impl<T, L, R> Differentiable<T> for Sub<L, R>
where
    T: Identifier,
    L: Differentiable<T>,
    R: Differentiable<T>,
{
    type Adjoint = Sub<L::Adjoint, R::Adjoint>;

    fn adjoint(&self, target: T) -> Self::Adjoint {
        let x = self.0.adjoint(target);
        let y = self.1.adjoint(target);

        Sub(x, y)
    }
}

impl<L: Node + std::fmt::Display, R: Node + std::fmt::Display> std::fmt::Display
    for Sub<L, R>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use crate::Stage::Evaluation;

        if Evaluation(&self.0).is_zero() == TFU::True {
            write!(f, "{}", self.1)
        } else if Evaluation(&self.1).is_zero() == TFU::True {
            write!(f, "{}", self.0)
        } else {
            write!(f, "({}) - ({})", self.0, self.1)
        }
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
#[derive(Copy, Clone, Debug, PartialEq, Contains)]
pub struct Mul<L, R>(#[op] pub L, #[op] pub R);

impl<L: Node, R: Node> Node for Mul<L, R> {
    fn is_zero(stage: Stage<&'_ Self>) -> TFU {
        stage.map(|node| &node.0).is_zero() | stage.map(|node| &node.1).is_zero()
    }

    fn is_one(stage: Stage<&'_ Self>) -> TFU {
        (stage.map(|node| &node.0).is_one() & stage.map(|node| &node.1).is_one())
            .true_or(TFU::Unknown)
    }
}

impl<D, F, L, LV, R, RV, OS, OC, OV> Function<D> for Mul<L, R>
where
    D: Database,
    F: Scalar,

    L: Function<D, Value = LV>,
    LV: Buffer<Field = F> + ZipMap<RV, Output<F> = OV>,

    R: Function<D, Value = RV>,
    RV: Buffer<Field = F> + ZipMap<LV, Output<F> = OV>,

    OS: Shape,
    OC: Class<OS, Buffer<F> = OV>,
    OV: Buffer<Field = F, Class = OC, Shape = OS>,

    LV::Shape: shapes::Zip<RV::Shape, Shape = OS>,
{
    type Error = Error<D, L, R>;
    type Value = OV;

    fn evaluate<DR: AsRef<D>>(&self, db: DR) -> Result<Self::Value, Self::Error> {
        use crate::Stage::Evaluation;

        let s0 = Evaluation(&self.0);
        let s1 = Evaluation(&self.1);

        // If either value is zero, we can just short-circuit...
        if (s0.is_zero() | s1.is_zero()).is_true() {
            return short_circuit!(@TakeNeither db[self.0, self.1]{F::zero()});
        }

        // If either value is unity, then we can also short-circuit...
        if s0.is_one().is_true() {
            // In this case, the left-hand value is one, i.e. we have 1 * x = x.
            return short_circuit!(@TakeRight db[self.0, self.1]);
        }

        if s1.is_one().is_true() {
            // In this case, the right-hand value is one, i.e. we have x * 1 = x.
            return short_circuit!(@TakeLeft db[self.0, self.1]);
        }

        // Otherwise we have to perform the multiplication...
        let x = self.0.evaluate(db.as_ref()).map_err(BinaryError::Left)?;
        let y = self.1.evaluate(db).map_err(BinaryError::Right)?;

        x.zip_map(y, |xi, yi| xi * yi).map_err(BinaryError::Output)
    }

    fn evaluate_shape<DR: AsRef<D>>(&self, db: DR) -> Result<ShapeOf<Self::Value>, Self::Error> {
        let l_shape = self.0.evaluate_shape(&db).map_err(BinaryError::Left)?;
        let r_shape = self.1.evaluate_shape(db).map_err(BinaryError::Right)?;

        l_shape.zip(r_shape).map_err(BinaryError::Output)
    }
}

impl<T, L, R> Differentiable<T> for Mul<L, R>
where
    T: Identifier,
    L: Differentiable<T> + Clone,
    R: Differentiable<T> + Clone,
{
    type Adjoint = Add<Mul<L::Adjoint, R>, Mul<R::Adjoint, L>>;

    fn adjoint(&self, target: T) -> Self::Adjoint {
        let gl = self.0.adjoint(target);
        let gr = self.1.adjoint(target);

        let ll = gl.mul(self.1.clone());
        let rr = gr.mul(self.0.clone());

        ll.add(rr)
    }
}

impl<L: Node + fmt::Display, R: Node + fmt::Display> fmt::Display for Mul<L, R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use crate::Stage::Evaluation;

        if Evaluation(&self.0).is_zero() == TFU::True {
            write!(f, "{}", self.1)
        } else if Evaluation(&self.1).is_zero() == TFU::True {
            write!(f, "{}", self.0)
        } else {
            write!(f, "({}) \u{2218} ({})", self.0, self.1)
        }
    }
}

/// Computes the (elementwise) quotient of two compatible [Buffer] types.
#[derive(Copy, Clone, Debug, PartialEq, Contains)]
pub struct Div<L, R>(#[op] pub L, #[op] pub R);

impl<L: Node, R: Node> Node for Div<L, R> {
    fn is_zero(stage: Stage<&'_ Self>) -> TFU { stage.map(|node| &node.0).is_zero() }

    fn is_one(stage: Stage<&'_ Self>) -> TFU {
        (stage.map(|node| &node.0).is_one() & stage.map(|node| &node.1).is_one())
            .true_or(TFU::Unknown)
    }
}

impl<D, F, L, LV, R, RV, OV> Function<D> for Div<L, R>
where
    D: Database,
    F: Scalar,

    L: Function<D, Value = LV>,
    LV: Buffer<Field = F> + ZipMap<RV, Output<F> = OV>,

    R: Function<D, Value = RV>,
    RV: Buffer<Field = F> + ZipMap<LV, Output<F> = OV>,

    OV: Buffer<Field = F>,

    LV::Shape: Zip<RV::Shape, Shape = OV::Shape>,
{
    type Error = Error<D, L, R>;
    type Value = OV;

    fn evaluate<DR: AsRef<D>>(&self, db: DR) -> Result<Self::Value, Self::Error> {
        use crate::Stage::Evaluation;

        // If the denominator takes value one, then we can skip the operation...
        if Evaluation(&self.1).is_one().is_true() {
            return short_circuit!(@TakeLeft db[self.0, self.1]);
        }

        let x = self.0.evaluate(db.as_ref()).map_err(BinaryError::Left)?;
        let y = self.1.evaluate(db).map_err(BinaryError::Right)?;

        x.zip_map(y, |xi, yi| xi / yi).map_err(BinaryError::Output)
    }

    fn evaluate_shape<DR: AsRef<D>>(&self, db: DR) -> Result<ShapeOf<Self::Value>, Self::Error> {
        let l_shape = self.0.evaluate_shape(&db).map_err(BinaryError::Left)?;
        let r_shape = self.1.evaluate_shape(db).map_err(BinaryError::Right)?;

        l_shape.zip(r_shape).map_err(BinaryError::Output)
    }
}

// impl<F, D, T, L, R> Differentiable<D, T> for Div<L, R>
// where
// F: Scalar,
// D: Database,
// T: Identifier,
// L: Differentiable<D, T>,
// R: Differentiable<D, T>,

// L::Value: Buffer<Field = F> + ZipMap<R::Value> + ZipMap<R::Adjoint,
// Output = L::Value>, R::Value: Buffer<Field = F> + ZipMap<L::Adjoint,
// Output = R::Value>,

// HadOut<L::Value, R::Value>: ZipMap<R::Value>,
// {
// type Adjoint = HadOut<HadOut<L::Value, R::Value>, R::Value>;

// fn grad(&self, db: &D, target: T) -> Result<Self::Adjoint, Self::Error> {
// let x = self.0.dual(db, target).map_err(BinaryError::Left)?;
// let y = self.1.dual(db, target).map_err(BinaryError::Right)?;

// let t1 = y.value.zip_map(&x.adjoint, |xi, yi| xi * yi)
// .unwrap();
// // .map_err(BinaryError::Output)?;
// let t2 = x.value.zip_map(&y.adjoint, |xi, yi| xi * yi)
// .unwrap();
// // .map_err(BinaryError::Output)?;

// let numerator = t2.zip_map(&t1, |yi, xi| xi - yi).unwrap();

// // t1.zip_map(&t2, |t1i, t2i| t1i + t2i).map_err(BinaryError::Output)
// Ok(numerator.zip_map(&y.value, |n, d| n / d).unwrap())
// }
// }

impl<T, L, R> Differentiable<T> for Div<L, R>
where
    T: Identifier,
    L: Differentiable<T> + Clone,
    R: Differentiable<T> + Clone,
{
    type Adjoint = Div<Sub<Mul<R, L::Adjoint>, Mul<L, R::Adjoint>>, crate::ops::Square<R>>;

    fn adjoint(&self, target: T) -> Self::Adjoint {
        let l = self.0.adjoint(target);
        let r = self.1.adjoint(target);

        let numerator = self.1.clone().mul(l).sub(self.0.clone().mul(r));

        numerator.div(self.1.clone().squared())
    }
}

impl<L: fmt::Display, R: Node + fmt::Display> fmt::Display for Div<L, R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}) / ({})", self.0, self.1)
    }
}
