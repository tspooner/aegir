use crate::{
    buffer::{Buffer, Field, FieldOf, OwnedOf},
    maths::{AddOut, MulOut},
    Compile,
    Contains,
    Database,
    Differentiable,
    Function,
    Identifier,
    Node,
};
use num_traits::{real::Real, Pow, Zero};
use std::{fmt, ops};

///////////////////////////////////////////////////////////////////////////////////////////////////
// Negate:
///////////////////////////////////////////////////////////////////////////////////////////////////
impl_unary!(
    /// Computes the additive inverse of a [Buffer].
    ///
    /// # Examples
    /// ```
    /// # #[macro_use] extern crate aegir;
    /// # use aegir::{Identifier, Differentiable, SimpleDatabase, Dual, maths::Negate};
    /// ids!(X::x);
    ///
    /// let f = Negate(X.to_var());
    ///
    /// assert_eq!(f.dual(&simple_db!(X => 1.0), X).unwrap(), dual!(-1.0, -1.0));
    /// assert_eq!(f.dual(&simple_db!(X => -1.0), X).unwrap(), dual!(1.0, -1.0));
    /// ```
    Negate["-{}"]: num_traits::real::Real,
    |x| { -x },
    |dx| { -dx }
);

impl<T, N> Compile<T> for Negate<N>
where
    T: Identifier,
    N: Compile<T>,
{
    type CompiledJacobian = Negate<N::CompiledJacobian>;
    type Error = N::Error;

    fn compile_grad(&self, target: T) -> Result<Self::CompiledJacobian, Self::Error> {
        self.0.compile_grad(target).map(Negate)
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Dirac:
///////////////////////////////////////////////////////////////////////////////////////////////////
fn dirac<F: Field + num_traits::Float>(x: F) -> F {
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
    /// # use aegir::{Identifier, Function, SimpleDatabase, Dual, maths::Dirac};
    /// # use std::f64::INFINITY;
    /// ids!(X::x);
    ///
    /// let f = Dirac(X.to_var());
    ///
    /// assert_eq!(f.evaluate(&simple_db!(X => -1.0)).unwrap(), 0.0);
    /// assert_eq!(f.evaluate(&simple_db!(X => 0.0)).unwrap(), INFINITY);
    /// assert_eq!(f.evaluate(&simple_db!(X => 1.0)).unwrap(), 0.0);
    /// ```
    Dirac["\u{03B4}({})"]: num_traits::Float,
    dirac,
    |_| { unimplemented!() }
);

///////////////////////////////////////////////////////////////////////////////////////////////////
// Sign:
///////////////////////////////////////////////////////////////////////////////////////////////////
/// Computes the sign of a [Buffer].
///
/// # Examples
/// ```
/// # #[macro_use] extern crate aegir;
/// # use aegir::{Identifier, Differentiable, SimpleDatabase, Dual, maths::Sign};
/// # use std::f64::INFINITY;
/// ids!(X::x);
///
/// let f = Sign(X.to_var());
///
/// assert_eq!(f.dual(&simple_db!(X => -1.0), X).unwrap(), dual!(-1.0, 0.0));
/// assert_eq!(f.dual(&simple_db!(X => 0.0), X).unwrap(), dual!(0.0, INFINITY));
/// assert_eq!(f.dual(&simple_db!(X => 1.0), X).unwrap(), dual!(1.0, 0.0));
/// ```
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Sign<N>(pub N);

impl<N: PartialEq> Node for Sign<N> {}

impl<T, N> Contains<T> for Sign<N>
where
    T: Identifier,
    N: Contains<T>,
{
    fn contains(&self, target: T) -> bool { self.0.contains(target) }
}

impl<D: Database, N: Function<D>> Function<D> for Sign<N>
where
    FieldOf<N::Codomain>: num_traits::real::Real,
{
    type Codomain = OwnedOf<N::Codomain>;
    type Error = N::Error;

    fn evaluate(&self, db: &D) -> Result<Self::Codomain, Self::Error> {
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

impl<D, T, N> Differentiable<D, T> for Sign<N>
where
    D: Database,
    T: Identifier,
    N: Differentiable<D, T>,

    FieldOf<N::Codomain>: num_traits::Float,
    OwnedOf<N::Codomain>: std::ops::Mul<OwnedOf<N::Jacobian>>,

    MulOut<OwnedOf<N::Codomain>, OwnedOf<N::Jacobian>>: Buffer<Field = FieldOf<N::Codomain>>,
{
    type Jacobian = MulOut<OwnedOf<N::Codomain>, OwnedOf<N::Jacobian>>;

    fn grad(&self, db: &D, target: T) -> Result<Self::Jacobian, Self::Error> {
        self.0
            .dual(db, target)
            .map(|d| d.value.map(dirac) * d.adjoint.map(|g| g))
    }
}

impl<X: fmt::Display + PartialEq> fmt::Display for Sign<X> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "sgn({})", self.0) }
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Absolute value:
///////////////////////////////////////////////////////////////////////////////////////////////////
/// Computes the absolute value of a [Buffer].
///
/// # Examples
/// ```
/// # #[macro_use] extern crate aegir;
/// # use aegir::{Identifier, Differentiable, SimpleDatabase, Dual, maths::Abs};
/// ids!(X::x);
///
/// let f = Abs(X.to_var());
///
/// assert_eq!(f.dual(&simple_db!(X => -1.0), X).unwrap(), dual!(1.0, -1.0));
/// assert_eq!(f.dual(&simple_db!(X => 0.0), X).unwrap(), dual!(0.0, 0.0));
/// assert_eq!(f.dual(&simple_db!(X => 1.0), X).unwrap(), dual!(1.0, 1.0));
/// ```
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Abs<N>(pub N);

impl<N: PartialEq> Node for Abs<N> {}

impl<T, N> Contains<T> for Abs<N>
where
    T: Identifier,
    N: Contains<T>,
{
    fn contains(&self, target: T) -> bool { self.0.contains(target) }
}

impl<D: Database, N: Function<D>> Function<D> for Abs<N>
where
    FieldOf<N::Codomain>: num_traits::real::Real,
{
    type Codomain = OwnedOf<N::Codomain>;
    type Error = N::Error;

    fn evaluate(&self, db: &D) -> Result<Self::Codomain, Self::Error> {
        self.0.evaluate(db).map(|buffer| buffer.map(|x| x.abs()))
    }
}

impl<D, T, N> Differentiable<D, T> for Abs<N>
where
    D: Database,
    T: Identifier,
    N: Differentiable<D, T>,

    FieldOf<N::Codomain>: num_traits::real::Real,
    OwnedOf<N::Codomain>: std::ops::Mul<N::Jacobian>,

    MulOut<OwnedOf<N::Codomain>, N::Jacobian>: Buffer<Field = FieldOf<N::Codomain>>,
{
    type Jacobian = MulOut<OwnedOf<N::Codomain>, N::Jacobian>;

    fn grad(&self, db: &D, target: T) -> Result<Self::Jacobian, Self::Error> {
        self.0.dual(db, target).map(|d| {
            d.value.map(|v| {
                if num_traits::Zero::is_zero(&v) {
                    v
                } else {
                    v.signum()
                }
            }) * d.adjoint
        })
    }
}

impl<X: fmt::Display + PartialEq> fmt::Display for Abs<X> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "|{}|", self.0) }
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Power:
///////////////////////////////////////////////////////////////////////////////////////////////////
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct SafeXlnX<N>(pub N);

impl<N: PartialEq> Node for SafeXlnX<N> {}

impl<T, N> Contains<T> for SafeXlnX<N>
where
    T: Identifier,
    N: Contains<T>,
{
    fn contains(&self, target: T) -> bool { self.0.contains(target) }
}

impl<D: Database, N: Function<D>> Function<D> for SafeXlnX<N>
where
    FieldOf<N::Codomain>: num_traits::real::Real,
{
    type Codomain = OwnedOf<N::Codomain>;
    type Error = N::Error;

    fn evaluate(&self, db: &D) -> Result<Self::Codomain, Self::Error> {
        self.0.evaluate(db).map(|buffer| {
            buffer.map(|x| {
                if x <= num_traits::zero() {
                    num_traits::zero()
                } else {
                    x * x.ln()
                }
            })
        })
    }
}

/// Computes the power of a [Buffer] to a [Field].
///
/// # Examples
/// ## x^2
/// ```
/// # #[macro_use] extern crate aegir;
/// # #[macro_use] extern crate ndarray;
/// # use aegir::{Identifier, Differentiable, SimpleDatabase, Dual, buffer::Buffer, maths::Power};
/// ids!(X::x);
///
/// let f = Power(X.to_var(), 2.0f64.to_constant());
///
/// assert_eq!(f.dual(&simple_db!(X => -1.0), X).unwrap(), dual!(1.0, -2.0));
/// assert_eq!(f.dual(&simple_db!(X => 0.0), X).unwrap(), dual!(0.0, 0.0));
/// assert_eq!(f.dual(&simple_db!(X => 1.0), X).unwrap(), dual!(1.0, 2.0));
/// assert_eq!(f.dual(&simple_db!(X => 2.0), X).unwrap(), dual!(4.0, 4.0));
/// ```
///
/// ## x^y
/// ```
/// # #[macro_use] extern crate aegir;
/// # #[macro_use] extern crate ndarray;
/// # use aegir::{Identifier, Differentiable, SimpleDatabase, Compile, Dual, buffer::Buffer, maths::Power};
/// # use aegir::{Function};
/// ids!(X::x, Y::y);
/// db!(DB { x: X, y: Y });
///
/// let f = Power(X.to_var(), Y.to_var());
///
/// assert!((
///     f.evaluate(&DB { x: 2.0, y: 1.5, }).unwrap() - 2.0f64.powf(1.5)
/// ) < 1e-5);
/// assert!((
///     f.grad(&DB { x: 2.0, y: 1.5, }, X).unwrap() - 1.5 * 2.0f64.powf(0.5)
/// ) < 1e-5);
/// assert!((
///     f.grad(&DB { x: 2.0, y: 1.5, }, Y).unwrap() - 2.0f64.powf(1.5) * 2.0f64.ln()
/// ) < 1e-5);
/// ```
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Power<N, E>(pub N, pub E);

impl<N: PartialEq, E: PartialEq> Node for Power<N, E> {}

impl<T, N, E> Contains<T> for Power<N, E>
where
    T: Identifier,
    N: Contains<T>,
    E: Contains<T>,
{
    fn contains(&self, target: T) -> bool { self.0.contains(target) || self.1.contains(target) }
}

impl<D, N, E> Function<D> for Power<N, E>
where
    D: Database,
    N: Function<D>,
    E: Function<D>,

    E::Codomain: Field,

    FieldOf<N::Codomain>: num_traits::Pow<E::Codomain, Output = FieldOf<N::Codomain>>,
{
    type Codomain = OwnedOf<N::Codomain>;
    type Error = either::Either<N::Error, E::Error>;

    fn evaluate(&self, db: &D) -> Result<Self::Codomain, Self::Error> {
        let exponent = self.1.evaluate(db).map_err(either::Right)?;

        self.0
            .evaluate(db)
            .map(move |buffer| buffer.map(|x| x.pow(exponent)))
            .map_err(either::Left)
    }
}

impl<D, T, N, E> Differentiable<D, T> for Power<N, E>
where
    D: Database,
    T: Identifier,
    N: Differentiable<D, T>,
    E: Differentiable<D, T>,

    E::Codomain: Field,
    N::Jacobian: std::ops::Mul<E::Codomain>,

    OwnedOf<N::Codomain>: std::ops::Mul<E::Codomain> + std::ops::Mul<E::Jacobian>,
    FieldOf<N::Codomain>:
        num_traits::real::Real + num_traits::Pow<E::Codomain, Output = FieldOf<N::Codomain>>,

    MulOut<N::Jacobian, E::Codomain>: std::ops::Add<MulOut<OwnedOf<N::Codomain>, E::Jacobian>>,
    MulOut<OwnedOf<N::Codomain>, E::Codomain>: Buffer<Field = FieldOf<Self::Codomain>>,
    MulOut<OwnedOf<N::Codomain>, E::Codomain>: std::ops::Mul<
        AddOut<MulOut<N::Jacobian, E::Codomain>, MulOut<OwnedOf<N::Codomain>, E::Jacobian>>,
    >,
    MulOut<
        MulOut<OwnedOf<N::Codomain>, E::Codomain>,
        AddOut<MulOut<N::Jacobian, E::Codomain>, MulOut<OwnedOf<N::Codomain>, E::Jacobian>>,
    >: Buffer<Field = FieldOf<Self::Codomain>>,
{
    type Jacobian = MulOut<
        MulOut<OwnedOf<N::Codomain>, E::Codomain>,
        AddOut<MulOut<N::Jacobian, E::Codomain>, MulOut<OwnedOf<N::Codomain>, E::Jacobian>>,
    >;

    fn grad(&self, db: &D, target: T) -> Result<Self::Jacobian, Self::Error> {
        let base = self.0.evaluate(db).map_err(either::Left)?;
        let base_grad = self.0.grad(db, target).map_err(either::Left)?;

        let exponent = self.1.evaluate(db).map_err(either::Right)?;
        let exponent_grad = self.1.grad(db, target).map_err(either::Right)?;

        let c: MulOut<OwnedOf<N::Codomain>, E::Codomain> =
            base.to_owned() * (exponent - num_traits::one());

        let t1: MulOut<N::Jacobian, E::Codomain> = base_grad * exponent;

        let t2: MulOut<OwnedOf<N::Codomain>, E::Jacobian> = base.map(|x| {
            if x <= num_traits::zero() {
                num_traits::zero()
            } else {
                x * x.ln()
            }
        }) * exponent_grad;

        let t: AddOut<
            MulOut<N::Jacobian, E::Codomain>,
            MulOut<OwnedOf<N::Codomain>, E::Jacobian>,
        > = t1 + t2;

        Ok(c * t)
    }
}

impl<T, N, E> Compile<T> for Power<N, E>
where
    T: Identifier,
    N: Compile<T> + Clone,
    E: Compile<T> + Clone,

    Self: Clone,
{
    type CompiledJacobian = Mul<
        Mul<N, SubOne<E>>,
        Add<Mul<N::CompiledJacobian, E>, Mul<SafeXlnX<N>, E::CompiledJacobian>>,
    >;
    type Error = either::Either<N::Error, E::Error>;

    fn compile_grad(&self, target: T) -> Result<Self::CompiledJacobian, Self::Error> {
        let base = self.0.clone();
        let base_g = self.0.compile_grad(target).map_err(|e| either::Left(e))?;

        let exponent = self.1.clone();
        let exponent_g = self.1.compile_grad(target).map_err(|e| either::Right(e))?;

        let c: Mul<N, SubOne<E>> = base.clone().mul(SubOne(exponent.clone()));
        let t1: Mul<N::CompiledJacobian, E> = base_g.mul(exponent);
        let t2: Mul<SafeXlnX<N>, E::CompiledJacobian> = SafeXlnX(base).mul(exponent_g);

        Ok(c.mul(t1.add(t2)))
    }
}

impl<X: fmt::Display, E: fmt::Display> fmt::Display for Power<X, E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({})^({})", self.0, self.1)
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Double:
///////////////////////////////////////////////////////////////////////////////////////////////////
/// Computes the double of a [Buffer].
///
/// # Examples
/// ```
/// # #[macro_use] extern crate aegir;
/// # #[macro_use] extern crate ndarray;
/// # use aegir::{Identifier, Differentiable, SimpleDatabase, Dual, maths::Double};
/// ids!(X::x);
///
/// let f = Double(X.to_var());
///
/// assert_eq!(f.dual(&simple_db!(X => -1.0), X).unwrap(), dual!(-2.0, 2.0));
/// assert_eq!(f.dual(&simple_db!(X => 0.0), X).unwrap(), dual!(0.0, 2.0));
/// assert_eq!(f.dual(&simple_db!(X => 1.0), X).unwrap(), dual!(2.0, 2.0));
/// ```
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Double<N>(pub N);

impl<N: PartialEq> Node for Double<N> {}

impl<T, N> Contains<T> for Double<N>
where
    T: Identifier,
    N: Contains<T>,
{
    fn contains(&self, target: T) -> bool { self.0.contains(target) }
}

impl<D, N> Function<D> for Double<N>
where
    D: Database,
    N: Function<D>,

    FieldOf<N::Codomain>: std::ops::Mul<FieldOf<N::Codomain>, Output = FieldOf<N::Codomain>>,
{
    type Codomain = OwnedOf<N::Codomain>;
    type Error = N::Error;

    fn evaluate(&self, db: &D) -> Result<Self::Codomain, Self::Error> {
        let two = num_traits::one::<FieldOf<N::Codomain>>() + num_traits::one();

        self.0.evaluate(db).map(|buffer| buffer.map(|x| two * x))
    }
}

impl<D, T, N> Differentiable<D, T> for Double<N>
where
    D: Database,
    T: Identifier,
    N: Differentiable<D, T>,

    FieldOf<N::Jacobian>: std::ops::Mul<FieldOf<N::Jacobian>, Output = FieldOf<N::Jacobian>>,
{
    type Jacobian = OwnedOf<N::Jacobian>;

    fn grad(&self, db: &D, target: T) -> Result<Self::Jacobian, Self::Error> {
        let two = num_traits::one::<FieldOf<N::Jacobian>>() + num_traits::one();

        self.0.dual(db, target).map(|d| d.adjoint.map(|g| two * g))
    }
}

impl<T, N> Compile<T> for Double<N>
where
    T: Identifier,
    N: Compile<T>,
{
    type CompiledJacobian = Double<N::CompiledJacobian>;
    type Error = N::Error;

    fn compile_grad(&self, target: T) -> Result<Self::CompiledJacobian, Self::Error> {
        self.0.compile_grad(target).map(Double)
    }
}

impl<X: fmt::Display> fmt::Display for Double<X> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "2({})", self.0) }
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Double:
///////////////////////////////////////////////////////////////////////////////////////////////////
/// Computes the square of a [Buffer].
///
/// # Examples
/// ```
/// # #[macro_use] extern crate aegir;
/// # #[macro_use] extern crate ndarray;
/// # use aegir::{Identifier, Differentiable, SimpleDatabase, Dual, maths::Square};
/// ids!(X::x);
///
/// let f = Square(X.to_var());
///
/// assert_eq!(f.dual(&simple_db!(X => -1.0), X).unwrap(), dual!(1.0, -2.0));
/// assert_eq!(f.dual(&simple_db!(X => 0.0), X).unwrap(), dual!(0.0, 0.0));
/// assert_eq!(f.dual(&simple_db!(X => 1.0), X).unwrap(), dual!(1.0, 2.0));
/// assert_eq!(f.dual(&simple_db!(X => 2.0), X).unwrap(), dual!(4.0, 4.0));
/// ```
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Square<N>(pub N);

impl<N: PartialEq> Node for Square<N> {}

impl<T, N> Contains<T> for Square<N>
where
    T: Identifier,
    N: Contains<T>,
{
    fn contains(&self, target: T) -> bool { self.0.contains(target) }
}

impl<D, N> Function<D> for Square<N>
where
    D: Database,
    N: Function<D>,

    FieldOf<N::Codomain>: num_traits::Pow<FieldOf<N::Codomain>, Output = FieldOf<N::Codomain>>,
{
    type Codomain = OwnedOf<N::Codomain>;
    type Error = N::Error;

    fn evaluate(&self, db: &D) -> Result<Self::Codomain, Self::Error> {
        let two = num_traits::one::<FieldOf<N::Codomain>>() + num_traits::one();

        self.0.evaluate(db).map(|buffer| buffer.map(|x| x.pow(two)))
    }
}

impl<D, T, N> Differentiable<D, T> for Square<N>
where
    D: Database,
    T: Identifier,
    N: Differentiable<D, T>,

    FieldOf<N::Codomain>: num_traits::Pow<FieldOf<N::Codomain>, Output = FieldOf<N::Codomain>>
        + std::ops::Mul<FieldOf<N::Codomain>, Output = FieldOf<N::Codomain>>,

    N::Codomain: std::ops::Mul<OwnedOf<N::Jacobian>>,

    MulOut<N::Codomain, OwnedOf<N::Jacobian>>: Buffer<Field = FieldOf<N::Codomain>>,
{
    type Jacobian = MulOut<N::Codomain, OwnedOf<N::Jacobian>>;

    fn grad(&self, db: &D, target: T) -> Result<Self::Jacobian, Self::Error> {
        let two = num_traits::one::<FieldOf<N::Jacobian>>() + num_traits::one();

        self.0
            .dual(db, target)
            .map(|d| d.value * d.adjoint.map(|g| two * g))
    }
}

impl<T, N> Compile<T> for Square<N>
where
    T: Identifier,
    N: Compile<T> + Clone,
{
    type CompiledJacobian = Mul<Double<N>, N::CompiledJacobian>;
    type Error = N::Error;

    fn compile_grad(&self, target: T) -> Result<Self::CompiledJacobian, Self::Error> {
        let g = self.0.compile_grad(target)?;

        Ok(Double(self.0.clone()).mul(g))
    }
}

impl<X: fmt::Display> fmt::Display for Square<X> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "({})^2", self.0) }
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Addition:
///////////////////////////////////////////////////////////////////////////////////////////////////
impl_trait!(
    @binary
    /// Computes the addition of two [Buffer] instances.
    ///
    /// # Examples
    /// ## x + y
    /// ```
    /// # #[macro_use] extern crate aegir;
    /// # #[macro_use] extern crate ndarray;
    /// # use aegir::{Identifier, Differentiable, SimpleDatabase, Dual, maths::Add};
    /// ids!(X::x, Y::y);
    /// db!(DB { x: X, y: Y });
    ///
    /// let f = Add(X.to_var(), Y.to_var());
    ///
    /// assert_eq!(f.dual(&DB { x: 1.0, y: 2.0, }, X).unwrap(), dual!(3.0, 1.0));
    /// assert_eq!(f.dual(&DB { x: 1.0, y: 2.0, }, Y).unwrap(), dual!(3.0, 1.0));
    /// ```
    ///
    /// ## x + y^2
    /// ```
    /// # #[macro_use] extern crate aegir;
    /// # #[macro_use] extern crate ndarray;
    /// # use aegir::{Identifier, Node, Differentiable, Dual, buffer::Buffer, maths::Add};
    /// ids!(X::x, Y::y);
    /// db!(DB { x: X, y: Y });
    ///
    /// let f = Add(X.to_var(), Y.to_var().pow(2.0f64.to_constant()));
    ///
    /// assert_eq!(f.dual(&DB { x: 1.0, y: 2.0, }, X).unwrap(), dual!(5.0, 1.0));
    /// assert_eq!(f.dual(&DB { x: 1.0, y: 2.0, }, Y).unwrap(), dual!(5.0, 4.0));
    /// ```
    Add["+"], ops::Add, |x, y| { x + y }, |dx, dy| { dx + dy }
);

impl<T, N1, N2> Compile<T> for Add<N1, N2>
where
    T: Identifier,
    N1: Compile<T>,
    N2: Compile<T>,
{
    type CompiledJacobian = Add<N1::CompiledJacobian, N2::CompiledJacobian>;
    type Error = either::Either<N1::Error, N2::Error>;

    fn compile_grad(&self, target: T) -> Result<Self::CompiledJacobian, Self::Error> {
        let gl = self.0.compile_grad(target).map_err(|e| either::Left(e))?;
        let gr = self.1.compile_grad(target).map_err(|e| either::Right(e))?;

        Ok(gl.add(gr))
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Addition (reduce):
///////////////////////////////////////////////////////////////////////////////////////////////////
/// Compute the sum over elements in a [Buffer].
///
/// # Examples
/// ```
/// # #[macro_use] extern crate aegir;
/// # #[macro_use] extern crate ndarray;
/// # use aegir::{Identifier, Differentiable, SimpleDatabase, Dual, maths::Reduce};
/// ids!(X::x);
///
/// let f = Reduce(X.to_var());
///
/// assert_eq!(
///     f.dual(&simple_db!(X => vec![1.0, 2.0, 3.0]), X).unwrap(),
///     dual!(6.0, vec![1.0, 1.0, 1.0])
/// );
/// ```
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Reduce<N>(pub N);

impl<N: PartialEq> Node for Reduce<N> {}

impl<T, N> Contains<T> for Reduce<N>
where
    T: Identifier,
    N: Contains<T>,
{
    fn contains(&self, target: T) -> bool { self.0.contains(target) }
}

impl<D, N> Function<D> for Reduce<N>
where
    D: Database,
    N: Function<D>,

    FieldOf<N::Codomain>: num_traits::Zero,
{
    type Codomain = FieldOf<N::Codomain>;
    type Error = N::Error;

    fn evaluate(&self, db: &D) -> Result<Self::Codomain, Self::Error> {
        self.0
            .evaluate(db)
            .map(|buffer| buffer.fold(num_traits::zero(), |acc, &x| acc + x))
    }
}

impl<D, T, N> Differentiable<D, T> for Reduce<N>
where
    D: Database,
    T: Identifier,
    N: Differentiable<D, T>,

    N::Jacobian: Buffer<Field = FieldOf<N::Codomain>>,

    FieldOf<N::Codomain>: num_traits::Zero,
{
    type Jacobian = N::Jacobian;

    fn grad(&self, db: &D, target: T) -> Result<Self::Jacobian, Self::Error> {
        self.0.grad(db, target)
    }
}

impl<T, N> Compile<T> for Reduce<N>
where
    T: Identifier,
    N: Compile<T>,
{
    type CompiledJacobian = N::CompiledJacobian;
    type Error = N::Error;

    fn compile_grad(&self, target: T) -> Result<Self::CompiledJacobian, Self::Error> {
        self.0.compile_grad(target)
    }
}

impl<N: fmt::Display> fmt::Display for Reduce<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "\u{03A3}_i ({})_i", self.0)
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Subtraction:
///////////////////////////////////////////////////////////////////////////////////////////////////
impl_trait!(
    @binary
    /// Computes the subtraction of two [Buffer] instances.
    ///
    /// # Examples
    /// ## x - y
    /// ```
    /// # #[macro_use] extern crate aegir;
    /// # #[macro_use] extern crate ndarray;
    /// # use aegir::{Identifier, Differentiable, SimpleDatabase, Dual, buffer::Buffer, maths::Sub};
    /// ids!(X::x, Y::y);
    /// db!(DB { x: X, y: Y });
    ///
    /// let f = Sub(X.to_var(), Y.to_var());
    ///
    /// assert_eq!(f.dual(&DB { x: 1.0, y: 2.0, }, X).unwrap(), dual!(-1.0, 1.0));
    /// assert_eq!(f.dual(&DB { x: 1.0, y: 2.0, }, Y).unwrap(), dual!(-1.0, -1.0));
    /// ```
    ///
    /// ## x - y^2
    /// ```
    /// # #[macro_use] extern crate aegir;
    /// # #[macro_use] extern crate ndarray;
    /// # use aegir::{Identifier, Node, Differentiable, Dual, buffer::Buffer, maths::Sub};
    /// ids!(X::x, Y::y);
    /// db!(DB { x: X, y: Y });
    ///
    /// let f = Sub(X.to_var(), Y.to_var().pow(2.0f64.to_constant()));
    ///
    /// assert_eq!(f.dual(&DB { x: 1.0, y: 2.0, }, X).unwrap(), dual!(-3.0, 1.0));
    /// assert_eq!(f.dual(&DB { x: 1.0, y: 2.0, }, Y).unwrap(), dual!(-3.0, -4.0));
    /// ```
    Sub["-"], ops::Sub, |x, y| { x - y }, |dx, dy| { dx - dy }
);

impl<T, N1, N2> Compile<T> for Sub<N1, N2>
where
    T: Identifier,
    N1: Compile<T>,
    N2: Compile<T>,
{
    type CompiledJacobian = Sub<N1::CompiledJacobian, N2::CompiledJacobian>;
    type Error = either::Either<N1::Error, N2::Error>;

    fn compile_grad(&self, target: T) -> Result<Self::CompiledJacobian, Self::Error> {
        let gl = self.0.compile_grad(target).map_err(|e| either::Left(e))?;
        let gr = self.1.compile_grad(target).map_err(|e| either::Right(e))?;

        Ok(gl.sub(gr))
    }
}

impl_unary!(
    /// Computes the subtraction of one from a [Buffer].
    ///
    /// # Examples
    /// ```
    /// # #[macro_use] extern crate aegir;
    /// # use aegir::{Identifier, Differentiable, SimpleDatabase, Dual, maths::SubOne};
    /// ids!(X::x);
    ///
    /// let f = SubOne(X.to_var());
    ///
    /// assert_eq!(f.dual(&simple_db!(X => 1.0), X).unwrap(), dual!(0.0, 1.0));
    /// assert_eq!(f.dual(&simple_db!(X => 0.0), X).unwrap(), dual!(-1.0, 1.0));
    /// assert_eq!(f.dual(&simple_db!(X => -1.0), X).unwrap(), dual!(-2.0, 1.0));
    /// ```
    SubOne["{} - 1"]: Field,
    |x| { x - num_traits::one() },
    |dx| { dx }
);

impl<T, N> Compile<T> for SubOne<N>
where
    T: Identifier,
    N: Compile<T>,
{
    type CompiledJacobian = N::CompiledJacobian;
    type Error = N::Error;

    fn compile_grad(&self, target: T) -> Result<Self::CompiledJacobian, Self::Error> {
        self.0.compile_grad(target)
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Multiplication:
///////////////////////////////////////////////////////////////////////////////////////////////////
/// Computes the multiplication of two [Buffer] instances.
///
/// # Examples
/// ## x . y
/// ```
/// # #[macro_use] extern crate aegir;
/// # #[macro_use] extern crate ndarray;
/// # use aegir::{Identifier, Differentiable, SimpleDatabase, Dual, buffer::Buffer, maths::Mul};
/// ids!(X::x, Y::y);
/// db!(DB { x: X, y: Y });
///
/// let f = Mul(X.to_var(), Y.to_var());
///
/// assert_eq!(f.dual(&DB { x: 3.0, y: 2.0, }, X).unwrap(), dual!(6.0, 2.0));
/// assert_eq!(f.dual(&DB { x: 3.0, y: 2.0, }, Y).unwrap(), dual!(6.0, 3.0));
/// ```
///
/// ## x . y^2
/// ```
/// # #[macro_use] extern crate aegir;
/// # #[macro_use] extern crate ndarray;
/// # use aegir::{Identifier, Node, Differentiable, Dual, buffer::Buffer, maths::Mul};
/// ids!(X::x, Y::y);
/// db!(DB { x: X, y: Y });
///
/// let f = Mul(X.to_var(), Y.to_var().pow(2.0f64.to_constant()));
///
/// assert_eq!(f.dual(&DB { x: 3.0, y: 2.0, }, X).unwrap(), dual!(12.0, 4.0));
/// assert_eq!(f.dual(&DB { x: 3.0, y: 2.0, }, Y).unwrap(), dual!(12.0, 12.0));
/// ```
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Mul<N1, N2>(pub N1, pub N2);

impl<N1: PartialEq, N2: PartialEq> Node for Mul<N1, N2> {}

impl<T, N1, N2> Contains<T> for Mul<N1, N2>
where
    T: Identifier,
    N1: Contains<T>,
    N2: Contains<T>,
{
    fn contains(&self, target: T) -> bool { self.0.contains(target) || self.1.contains(target) }
}

impl<D, N1, N2> Function<D> for Mul<N1, N2>
where
    D: Database,
    N1: Function<D>,
    N2: Function<D>,

    N1::Codomain: ops::Mul<N2::Codomain>,

    MulOut<N1::Codomain, N2::Codomain>: Buffer,
{
    type Codomain = MulOut<N1::Codomain, N2::Codomain>;
    type Error = either::Either<N1::Error, N2::Error>;

    fn evaluate(&self, db: &D) -> Result<Self::Codomain, Self::Error> {
        self.0
            .evaluate(db)
            .map_err(either::Either::Left)
            .and_then(|x| {
                self.1
                    .evaluate(db)
                    .map(|y| x * y)
                    .map_err(either::Either::Right)
            })
    }
}

impl<D, T, N1, N2> Differentiable<D, T> for Mul<N1, N2>
where
    D: Database,
    T: Identifier,
    N1: Differentiable<D, T>,
    N2: Differentiable<D, T>,

    N1::Codomain: ops::Mul<N2::Codomain>,
    N2::Jacobian: ops::Mul<N1::Codomain>,
    N1::Jacobian: ops::Mul<N2::Codomain>,

    for<'a> &'a N1::Codomain:
        ops::Mul<&'a N2::Codomain, Output = MulOut<N1::Codomain, N2::Codomain>>,

    MulOut<N1::Codomain, N2::Codomain>: Buffer,
    MulOut<N1::Jacobian, N2::Codomain>: ops::Add<MulOut<N2::Jacobian, N1::Codomain>>,

    AddOut<MulOut<N1::Jacobian, N2::Codomain>, MulOut<N2::Jacobian, N1::Codomain>>:
        Buffer<Field = FieldOf<MulOut<N1::Codomain, N2::Codomain>>>,
{
    type Jacobian = <MulOut<N1::Jacobian, N2::Codomain> as ops::Add<
        MulOut<N2::Jacobian, N1::Codomain>,
    >>::Output;

    fn grad(&self, db: &D, target: T) -> Result<Self::Jacobian, Self::Error> {
        let d1 = self.0.dual(db, target).map_err(either::Either::Left)?;
        let d2 = self.1.dual(db, target).map_err(either::Either::Right)?;

        Ok(d1.adjoint * d2.value + d2.adjoint * d1.value)
    }

    fn dual(
        &self,
        db: &D,
        target: T,
    ) -> Result<crate::Dual<Self::Codomain, Self::Jacobian>, Self::Error> {
        let d1 = self.0.dual(db, target).map_err(either::Either::Left)?;
        let d2 = self.1.dual(db, target).map_err(either::Either::Right)?;

        Ok(crate::Dual {
            value: &d1.value * &d2.value,
            adjoint: d1.adjoint * d2.value + d2.adjoint * d1.value,
        })
    }
}

impl<T, N1, N2> Compile<T> for Mul<N1, N2>
where
    T: Identifier,
    N1: Compile<T> + Clone,
    N2: Compile<T> + Clone,
{
    type CompiledJacobian = Add<Mul<N1::CompiledJacobian, N2>, Mul<N2::CompiledJacobian, N1>>;
    type Error = either::Either<N1::Error, N2::Error>;

    fn compile_grad(&self, target: T) -> Result<Self::CompiledJacobian, Self::Error> {
        let gl = self.0.compile_grad(target).map_err(|e| either::Left(e))?;
        let gr = self.1.compile_grad(target).map_err(|e| either::Right(e))?;

        let ll = gl.mul(self.1.clone());
        let rr = gr.mul(self.0.clone());

        Ok(ll.add(rr))
    }
}

impl<N1: fmt::Display, N2: Node + fmt::Display> fmt::Display for Mul<N1, N2> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "{} {}", self.0, self.1) }
}
