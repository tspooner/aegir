use crate::{
    buffer::{Buffer, Field, FieldOf},
    maths::{AddOut, Mul, Reduce},
    Compile,
    Contains,
    Database,
    Differentiable,
    Function,
    Identifier,
    Node,
};
use ndarray::{ArrayBase, Ix1, Ix2};

#[derive(Debug)]
pub enum LinalgError<E1, E2> {
    Inner1(E1),
    Inner2(E2),
    IncompatibleVectors(usize, usize),
    IncompatibleMatrices((usize, usize), (usize, usize)),
}

impl<E1, E2> std::fmt::Display for LinalgError<E1, E2>
where
    E1: std::fmt::Display,
    E2: std::fmt::Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LinalgError::Inner1(e) => e.fmt(f),
            LinalgError::Inner2(e) => e.fmt(f),
            LinalgError::IncompatibleVectors(a, b) => {
                write!(f, "Lengths of vectors are incompatible: {} vs {}.", a, b)
            },
            LinalgError::IncompatibleMatrices(a, b) => write!(
                f,
                "Matrices are of incompatible shapes: {:?} vs {:?}.",
                a, b
            ),
        }
    }
}

impl<E1, E2> std::error::Error for LinalgError<E1, E2>
where
    E1: std::fmt::Debug + std::fmt::Display,
    E2: std::fmt::Debug + std::fmt::Display,
{
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Inner Products:
///////////////////////////////////////////////////////////////////////////////////////////////////
impl_newtype!(
    /// Computes the inner product of two vector [Buffers](Buffer).
    ///
    /// # Examples
    /// ```
    /// # #[macro_use] extern crate aegir;
    /// # #[macro_use] extern crate ndarray;
    /// # use aegir::{Identifier, Differentiable, SimpleDatabase, Dual, maths::InnerProduct};
    /// ids!(X::x, Y::y);
    /// db!(DB { x: X, y: Y });
    ///
    /// let f = InnerProduct::new(X.to_var(), Y.to_var());
    /// let db = DB {
    ///     x: array![1.0, 2.0, 3.0],
    ///     y: array![-1.0, 0.0, 2.0]
    /// };
    ///
    /// assert_eq!(f.dual(&db, X).unwrap(), dual!(5.0, array![-1.0, 0.0, 2.0]));
    /// assert_eq!(f.dual(&db, Y).unwrap(), dual!(5.0, array![1.0, 2.0, 3.0]));
    /// ```
    InnerProduct<N1, N2>(Reduce<Mul<N1, N2>>)
);

impl<N1, N2> InnerProduct<N1, N2> {
    pub fn new(n1: N1, n2: N2) -> Self { InnerProduct(Reduce(Mul(n1, n2))) }
}

impl<T, N1, N2> Compile<T> for InnerProduct<N1, N2>
where
    T: Identifier,

    Reduce<Mul<N1, N2>>: Compile<T>,
{
    type CompiledJacobian = <Reduce<Mul<N1, N2>> as Compile<T>>::CompiledJacobian;
    type Error = <Reduce<Mul<N1, N2>> as Compile<T>>::Error;

    fn compile_grad(&self, target: T) -> Result<Self::CompiledJacobian, Self::Error> {
        self.0.compile_grad(target)
    }
}

impl<N1: std::fmt::Display, N2: std::fmt::Display> std::fmt::Display for InnerProduct<N1, N2> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "\u{27E8}{}, {}\u{27E9}", self.0 .0 .0, self.0 .0 .1)
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Outer Products:
///////////////////////////////////////////////////////////////////////////////////////////////////
pub trait OuterProductTrait<T>: Buffer
where
    T: Buffer<Field = Self::Field>,
{
    type Output: Buffer;

    fn outer_product<E1, E2>(&self, rhs: &T) -> Result<Self::Output, LinalgError<E1, E2>>;
}

impl<F, D> OuterProductTrait<ArrayBase<D, Ix1>> for ArrayBase<D, Ix1>
where
    F: Field + 'static,
    D: ndarray::Data<Elem = F> + ndarray::RawDataClone,
{
    type Output = ndarray::Array2<F>;

    fn outer_product<E1, E2>(
        &self,
        rhs: &ArrayBase<D, Ix1>,
    ) -> Result<ndarray::Array2<F>, LinalgError<E1, E2>> {
        let nx = self.len();
        let ny = rhs.len();

        self.view()
            .into_shape((nx, 1))
            .unwrap()
            .mat_mul(&rhs.view().into_shape((1, ny)).unwrap())
    }
}

impl<F> OuterProductTrait<Vec<F>> for Vec<F>
where
    F: Field + 'static,
{
    type Output = ndarray::Array2<F>;

    fn outer_product<E1, E2>(
        &self,
        rhs: &Vec<F>,
    ) -> Result<ndarray::Array2<F>, LinalgError<E1, E2>> {
        ndarray::aview1(self).outer_product(&ndarray::aview1(rhs))
    }
}

/// Computes the outer product of two vector [Buffers](Buffer).
///
/// # Examples
/// ```
/// # #[macro_use] extern crate aegir;
/// # #[macro_use] extern crate ndarray;
/// # use aegir::{Identifier, Function, Differentiable, SimpleDatabase, Dual, maths::OuterProduct};
/// ids!(X::x, Y::y);
/// db!(DB { x: X, y: Y });
///
/// let f = OuterProduct(X.to_var(), Y.to_var());
/// let db = DB {
///     x: array![1.0, 2.0, 3.0],
///     y: array![-1.0, 0.0, 2.0]
/// };
///
/// assert_eq!(f.evaluate(&db).unwrap(), array![
///     [-1.0, 0.0, 2.0],
///     [-2.0, 0.0, 4.0],
///     [-3.0, 0.0, 6.0]
/// ]);
/// assert_eq!(f.grad(&db, X).unwrap(), array![
///     [-1.0, 0.0, 2.0],
///     [-1.0, 0.0, 2.0],
///     [-1.0, 0.0, 2.0]
/// ]);
/// assert_eq!(f.grad(&db, Y).unwrap(), array![
///     [1.0, 1.0, 1.0],
///     [2.0, 2.0, 2.0],
///     [3.0, 3.0, 3.0]
/// ]);
/// ```
pub struct OuterProduct<N1, N2>(pub N1, pub N2);

impl<N1, N2> Node for OuterProduct<N1, N2> {}

impl<T, N1, N2> Contains<T> for OuterProduct<N1, N2>
where
    T: Identifier,
    N1: Contains<T>,
    N2: Contains<T>,
{
    fn contains(&self, target: T) -> bool { self.0.contains(target) || self.1.contains(target) }
}

impl<D, N1, N2> Function<D> for OuterProduct<N1, N2>
where
    D: Database,
    N1: Function<D>,
    N2: Function<D>,

    N1::Codomain: OuterProductTrait<N2::Codomain>,
    N2::Codomain: Buffer<Field = FieldOf<N1::Codomain>>,
{
    type Codomain = <N1::Codomain as OuterProductTrait<N2::Codomain>>::Output;
    type Error = LinalgError<N1::Error, N2::Error>;

    fn evaluate(&self, db: &D) -> Result<Self::Codomain, Self::Error> {
        let x = self.0.evaluate(db).map_err(|e| LinalgError::Inner1(e))?;
        let y = self.1.evaluate(db).map_err(|e| LinalgError::Inner2(e))?;

        x.outer_product(&y)
    }
}

impl<D, T, N1, N2> Differentiable<D, T> for OuterProduct<N1, N2>
where
    D: Database,
    T: Identifier,
    N1: Differentiable<D, T>,
    N2: Differentiable<D, T>,

    N1::Codomain: OuterProductTrait<N2::Codomain>,
    N2::Codomain: Buffer<Field = FieldOf<N1::Codomain>>,

    N1::Jacobian: OuterProductTrait<N2::Codomain>,
    N1::Codomain: OuterProductTrait<N2::Jacobian>,

    <N1::Jacobian as OuterProductTrait<N2::Codomain>>::Output:
        std::ops::Add<<N1::Codomain as OuterProductTrait<N2::Jacobian>>::Output>,

    AddOut<
        <N1::Jacobian as OuterProductTrait<N2::Codomain>>::Output,
        <N1::Codomain as OuterProductTrait<N2::Jacobian>>::Output,
    >: Buffer<Field = FieldOf<Self::Codomain>>,
{
    type Jacobian = AddOut<
        <N1::Jacobian as OuterProductTrait<N2::Codomain>>::Output,
        <N1::Codomain as OuterProductTrait<N2::Jacobian>>::Output,
    >;

    fn grad(&self, db: &D, target: T) -> Result<Self::Jacobian, Self::Error> {
        let dual_x = self
            .0
            .dual(db, target)
            .map_err(|e| LinalgError::Inner1(e))?;
        let dual_y = self
            .1
            .dual(db, target)
            .map_err(|e| LinalgError::Inner2(e))?;

        dual_x
            .adjoint
            .outer_product(&dual_y.value)
            .and_then(|x| dual_x.value.outer_product(&dual_y.adjoint).map(|y| x + y))
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Matrix Multiplication
///////////////////////////////////////////////////////////////////////////////////////////////////
pub trait MatMulTrait<T>: Buffer
where
    T: Buffer<Field = Self::Field>,
{
    type Output: Buffer;

    fn mat_mul<E1, E2>(&self, rhs: &T) -> Result<Self::Output, LinalgError<E1, E2>>;
}

impl<F, D> MatMulTrait<ArrayBase<D, Ix2>> for ArrayBase<D, Ix2>
where
    F: Field + 'static,
    D: ndarray::Data<Elem = F> + ndarray::RawDataClone,
{
    type Output = ndarray::Array2<F>;

    fn mat_mul<E1, E2>(
        &self,
        rhs: &ArrayBase<D, Ix2>,
    ) -> Result<ndarray::Array2<F>, LinalgError<E1, E2>> {
        if self.ncols() == rhs.nrows() {
            Ok(self.dot(rhs))
        } else {
            Err(LinalgError::IncompatibleMatrices(self.dim(), rhs.dim()))
        }
    }
}

/// Computes the product of two matrix [Buffers](Buffer).
///
/// # Examples
/// ```
/// # #[macro_use] extern crate aegir;
/// # #[macro_use] extern crate ndarray;
/// # use aegir::{Identifier, Function, Differentiable, SimpleDatabase, Dual, maths::MatMul};
/// ids!(X::x, Y::y);
/// db!(DB { x: X, y: Y });
///
/// let f = MatMul(X.to_var(), Y.to_var());
/// let db = DB {
///     x: array![
///         [0.0, -1.0],
///         [1.0, 0.0]
///     ],
///     y: array![
///         [1.0, 2.0],
///         [3.0, 4.0]
///     ]
/// };
///
/// assert_eq!(f.evaluate(&db).unwrap(), array![
///     [-3.0, -4.0],
///     [1.0, 2.0]
/// ]);
/// assert_eq!(f.grad(&db, X).unwrap(), array![
///     [4.0, 6.0],
///     [4.0, 6.0]
/// ]);
/// assert_eq!(f.grad(&db, Y).unwrap(), array![
///     [-1.0, -1.0],
///     [1.0, 1.0]
/// ]);
/// ```
pub struct MatMul<N1, N2>(pub N1, pub N2);

impl<N1, N2> Node for MatMul<N1, N2> {}

impl<T, N1, N2> Contains<T> for MatMul<N1, N2>
where
    T: Identifier,
    N1: Contains<T>,
    N2: Contains<T>,
{
    fn contains(&self, target: T) -> bool { self.0.contains(target) || self.1.contains(target) }
}

impl<D, N1, N2> Function<D> for MatMul<N1, N2>
where
    D: Database,
    N1: Function<D>,
    N2: Function<D>,

    N1::Codomain: MatMulTrait<N2::Codomain>,
    N2::Codomain: Buffer<Field = <N1::Codomain as Buffer>::Field>,
{
    type Codomain = <N1::Codomain as MatMulTrait<N2::Codomain>>::Output;
    type Error = LinalgError<N1::Error, N2::Error>;

    fn evaluate(&self, db: &D) -> Result<Self::Codomain, Self::Error> {
        let x = self.0.evaluate(db).map_err(|e| LinalgError::Inner1(e))?;
        let y = self.1.evaluate(db).map_err(|e| LinalgError::Inner2(e))?;

        x.mat_mul(&y)
    }
}

impl<D, T, N1, N2> Differentiable<D, T> for MatMul<N1, N2>
where
    D: Database,
    T: Identifier,
    N1: Differentiable<D, T>,
    N2: Differentiable<D, T>,

    N1::Codomain: MatMulTrait<N2::Codomain>,
    N2::Codomain: Buffer<Field = FieldOf<N1::Codomain>>,

    N1::Jacobian: MatMulTrait<N2::Codomain>,
    N1::Codomain: MatMulTrait<N2::Jacobian>,

    <N1::Jacobian as MatMulTrait<N2::Codomain>>::Output:
        std::ops::Add<<N1::Codomain as MatMulTrait<N2::Jacobian>>::Output>,

    AddOut<
        <N1::Jacobian as MatMulTrait<N2::Codomain>>::Output,
        <N1::Codomain as MatMulTrait<N2::Jacobian>>::Output,
    >: Buffer<Field = FieldOf<Self::Codomain>>,
{
    type Jacobian = AddOut<
        <N1::Jacobian as MatMulTrait<N2::Codomain>>::Output,
        <N1::Codomain as MatMulTrait<N2::Jacobian>>::Output,
    >;

    fn grad(&self, db: &D, target: T) -> Result<Self::Jacobian, Self::Error> {
        let dual_x = self
            .0
            .dual(db, target)
            .map_err(|e| LinalgError::Inner1(e))?;
        let dual_y = self
            .1
            .dual(db, target)
            .map_err(|e| LinalgError::Inner2(e))?;

        dual_x
            .adjoint
            .mat_mul(&dual_y.value)
            .and_then(|x| dual_x.value.mat_mul(&dual_y.adjoint).map(|y| x + y))
    }
}
