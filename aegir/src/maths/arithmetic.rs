use crate::{
    buffer::{Buffer, Field, FieldOf, OwnedOf},
    maths::{AddOut, MulOut},
    sources::Constant,
    Compile,
    Contains,
    Database,
    Differentiable,
    Function,
    Identifier,
    Node,
};
use num_traits::{real::Real, Pow};
use std::{fmt, ops};

///////////////////////////////////////////////////////////////////////////////////////////////////
// Negate:
///////////////////////////////////////////////////////////////////////////////////////////////////
impl_unary!(
    /// Computes the element-wise additive inverse of a [Buffer].
    ///
    /// # Examples
    /// ```
    /// # #[macro_use] extern crate aegir;
    /// # use aegir::{Identifier, Differentiable, SimpleDatabase, dual::Dual, maths::Negate};
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
    /// Computes the element-wise dirac delta about zero of a [Buffer].
    ///
    /// # Examples
    /// ```
    /// # #[macro_use] extern crate aegir;
    /// # use aegir::{Identifier, Function, SimpleDatabase, dual::Dual, maths::Dirac};
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
/// Computes the element-wise sign of a [Buffer].
///
/// # Examples
/// ```
/// # #[macro_use] extern crate aegir;
/// # use aegir::{Identifier, Differentiable, SimpleDatabase, dual::Dual, maths::Sign};
/// # use std::f64::INFINITY;
/// ids!(X::x);
///
/// let f = Sign(X.to_var());
///
/// assert_eq!(f.dual(&simple_db!(X => -1.0), X).unwrap(), dual!(-1.0, 0.0));
/// assert_eq!(f.dual(&simple_db!(X => 0.0), X).unwrap(), dual!(0.0, INFINITY));
/// assert_eq!(f.dual(&simple_db!(X => 1.0), X).unwrap(), dual!(1.0, 0.0));
/// ```
#[derive(Copy, Clone, Debug)]
pub struct Sign<N>(pub N);

impl<N> Node for Sign<N> {}

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

impl<X: fmt::Display> fmt::Display for Sign<X> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "sgn({})", self.0) }
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Absolute value:
///////////////////////////////////////////////////////////////////////////////////////////////////
/// Computes the element-wise absolute value of a [Buffer].
///
/// # Examples
/// ```
/// # #[macro_use] extern crate aegir;
/// # use aegir::{Identifier, Differentiable, SimpleDatabase, dual::Dual, maths::Abs};
/// ids!(X::x);
///
/// let f = Abs(X.to_var());
///
/// assert_eq!(f.dual(&simple_db!(X => -1.0), X).unwrap(), dual!(1.0, -1.0));
/// assert_eq!(f.dual(&simple_db!(X => 0.0), X).unwrap(), dual!(0.0, 0.0));
/// assert_eq!(f.dual(&simple_db!(X => 1.0), X).unwrap(), dual!(1.0, 1.0));
/// ```
#[derive(Copy, Clone, Debug)]
pub struct Abs<N>(pub N);

impl<N> Node for Abs<N> {}

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

impl<X: fmt::Display> fmt::Display for Abs<X> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "|{}|", self.0) }
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Power:
///////////////////////////////////////////////////////////////////////////////////////////////////
/// Computes the element-wise power of a [Buffer] to a [Field].
///
/// # Examples
/// ```
/// # #[macro_use] extern crate aegir;
/// # #[macro_use] extern crate ndarray;
/// # use aegir::{Identifier, Differentiable, SimpleDatabase, dual::Dual, maths::Power};
/// ids!(X::x);
///
/// let f = Power(X.to_var(), 2.0);
///
/// assert_eq!(f.dual(&simple_db!(X => -1.0), X).unwrap(), dual!(1.0, -2.0));
/// assert_eq!(f.dual(&simple_db!(X => 0.0), X).unwrap(), dual!(0.0, 0.0));
/// assert_eq!(f.dual(&simple_db!(X => 1.0), X).unwrap(), dual!(1.0, 2.0));
/// assert_eq!(f.dual(&simple_db!(X => 2.0), X).unwrap(), dual!(4.0, 4.0));
/// ```
#[derive(Copy, Clone, Debug)]
pub struct Power<N, P>(pub N, pub P);

impl<N, P> Node for Power<N, P> {}

impl<T, N, P> Contains<T> for Power<N, P>
where
    T: Identifier,
    N: Contains<T>,
{
    fn contains(&self, target: T) -> bool { self.0.contains(target) }
}

impl<D, N, P> Function<D> for Power<N, P>
where
    D: Database,
    N: Function<D>,
    P: Field,

    FieldOf<N::Codomain>: num_traits::Pow<P, Output = FieldOf<N::Codomain>>,
{
    type Codomain = OwnedOf<N::Codomain>;
    type Error = N::Error;

    fn evaluate(&self, db: &D) -> Result<Self::Codomain, Self::Error> {
        self.0
            .evaluate(db)
            .map(|buffer| buffer.map(|x| x.pow(self.1.clone())))
    }
}

impl<D, T, N, P> Differentiable<D, T> for Power<N, P>
where
    D: Database,
    T: Identifier,
    N: Differentiable<D, T>,
    P: Field + num_traits::One + std::ops::Sub<P, Output = P>,

    FieldOf<N::Codomain>: num_traits::Pow<P, Output = FieldOf<N::Codomain>>,
    FieldOf<N::Jacobian>: std::ops::Mul<P, Output = FieldOf<N::Jacobian>>,
    OwnedOf<N::Codomain>: std::ops::Mul<OwnedOf<N::Jacobian>>,

    MulOut<OwnedOf<N::Codomain>, OwnedOf<N::Jacobian>>: Buffer<Field = FieldOf<N::Codomain>>,
{
    type Jacobian = MulOut<OwnedOf<N::Codomain>, OwnedOf<N::Jacobian>>;

    fn grad(&self, db: &D, target: T) -> Result<Self::Jacobian, Self::Error> {
        let np = self.1 - num_traits::one();

        self.0
            .dual(db, target)
            .map(|d| d.value.map(|v| v.pow(np)) * d.adjoint.map(|g| g * self.1))
    }
}

impl<T, N, P> Compile<T> for Power<N, P>
where
    T: Identifier,
    N: Compile<T> + Clone,
    P: Field + num_traits::One,
{
    type CompiledJacobian = Mul<Mul<Constant<P>, Power<N, P>>, N::CompiledJacobian>;
    type Error = N::Error;

    fn compile_grad(&self, target: T) -> Result<Self::CompiledJacobian, Self::Error> {
        let c = Constant(self.1);
        let n = self.0.clone().pow(self.1 - num_traits::one());
        let g = self.0.compile_grad(target)?;

        Ok(c.mul(n).mul(g))
    }
}

impl<X: fmt::Display, P: fmt::Display> fmt::Display for Power<X, P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({})^{}", self.0, self.1)
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Double:
///////////////////////////////////////////////////////////////////////////////////////////////////
/// Computes the element-wise double of a [Buffer].
///
/// # Examples
/// ```
/// # #[macro_use] extern crate aegir;
/// # #[macro_use] extern crate ndarray;
/// # use aegir::{Identifier, Differentiable, SimpleDatabase, dual::Dual, maths::Double};
/// ids!(X::x);
///
/// let f = Double(X.to_var());
///
/// assert_eq!(f.dual(&simple_db!(X => -1.0), X).unwrap(), dual!(-2.0, 2.0));
/// assert_eq!(f.dual(&simple_db!(X => 0.0), X).unwrap(), dual!(0.0, 2.0));
/// assert_eq!(f.dual(&simple_db!(X => 1.0), X).unwrap(), dual!(2.0, 2.0));
/// ```
#[derive(Copy, Clone, Debug)]
pub struct Double<N>(pub N);

impl<N> Node for Double<N> {}

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

    FieldOf<N::Codomain>:
        num_traits::One + std::ops::Mul<FieldOf<N::Codomain>, Output = FieldOf<N::Codomain>>,
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

    FieldOf<N::Jacobian>:
        num_traits::One + std::ops::Mul<FieldOf<N::Jacobian>, Output = FieldOf<N::Jacobian>>,
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
/// Computes the element-wise square of a [Buffer].
///
/// # Examples
/// ```
/// # #[macro_use] extern crate aegir;
/// # #[macro_use] extern crate ndarray;
/// # use aegir::{Identifier, Differentiable, SimpleDatabase, dual::Dual, maths::Square};
/// ids!(X::x);
///
/// let f = Square(X.to_var());
///
/// assert_eq!(f.dual(&simple_db!(X => -1.0), X).unwrap(), dual!(1.0, -2.0));
/// assert_eq!(f.dual(&simple_db!(X => 0.0), X).unwrap(), dual!(0.0, 0.0));
/// assert_eq!(f.dual(&simple_db!(X => 1.0), X).unwrap(), dual!(1.0, 2.0));
/// assert_eq!(f.dual(&simple_db!(X => 2.0), X).unwrap(), dual!(4.0, 4.0));
/// ```
#[derive(Copy, Clone, Debug)]
pub struct Square<N>(pub N);

impl<N> Node for Square<N> {}

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

    FieldOf<N::Codomain>:
        num_traits::One + num_traits::Pow<FieldOf<N::Codomain>, Output = FieldOf<N::Codomain>>,
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

    FieldOf<N::Codomain>: num_traits::One
        + num_traits::Pow<FieldOf<N::Codomain>, Output = FieldOf<N::Codomain>>
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
    /// Computes the element-wise addition of two [Buffer] instances.
    ///
    /// # Examples
    /// ## x + y
    /// ```
    /// # #[macro_use] extern crate aegir;
    /// # #[macro_use] extern crate ndarray;
    /// # use aegir::{Identifier, Differentiable, SimpleDatabase, dual::Dual, maths::Add};
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
    /// # use aegir::{Identifier, Node, Differentiable, dual::Dual, maths::Add};
    /// ids!(X::x, Y::y);
    /// db!(DB { x: X, y: Y });
    ///
    /// let f = Add(X.to_var(), Y.to_var().pow(2.0));
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
// Subtraction:
///////////////////////////////////////////////////////////////////////////////////////////////////
impl_trait!(
    @binary
    /// Computes the element-wise subtraction of two [Buffer] instances.
    ///
    /// # Examples
    /// ## x - y
    /// ```
    /// # #[macro_use] extern crate aegir;
    /// # #[macro_use] extern crate ndarray;
    /// # use aegir::{Identifier, Differentiable, SimpleDatabase, dual::Dual, maths::Sub};
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
    /// # use aegir::{Identifier, Node, Differentiable, dual::Dual, maths::Sub};
    /// ids!(X::x, Y::y);
    /// db!(DB { x: X, y: Y });
    ///
    /// let f = Sub(X.to_var(), Y.to_var().pow(2.0));
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

/// Computes the element-wise multiplication of two [Buffer] instances.
///
/// # Examples
/// ## x . y
/// ```
/// # #[macro_use] extern crate aegir;
/// # #[macro_use] extern crate ndarray;
/// # use aegir::{Identifier, Differentiable, SimpleDatabase, dual::Dual, maths::Mul};
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
/// # use aegir::{Identifier, Node, Differentiable, dual::Dual, maths::Mul};
/// ids!(X::x, Y::y);
/// db!(DB { x: X, y: Y });
///
/// let f = Mul(X.to_var(), Y.to_var().pow(2.0));
///
/// assert_eq!(f.dual(&DB { x: 3.0, y: 2.0, }, X).unwrap(), dual!(12.0, 4.0));
/// assert_eq!(f.dual(&DB { x: 3.0, y: 2.0, }, Y).unwrap(), dual!(12.0, 12.0));
/// ```
#[derive(Copy, Clone, Debug)]
pub struct Mul<N1, N2>(pub N1, pub N2);

impl<N1, N2> Node for Mul<N1, N2> {}

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
    ) -> Result<crate::dual::Dual<Self::Codomain, Self::Jacobian>, Self::Error> {
        let d1 = self.0.dual(db, target).map_err(either::Either::Left)?;
        let d2 = self.1.dual(db, target).map_err(either::Either::Right)?;

        Ok(crate::dual::Dual {
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
