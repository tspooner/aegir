use crate::{
    Identifier, State, Node, Contains, Function, Differentiable,
    buffer::{Buffer, ScalarBuffer, FieldOf},
    ops::{AddOut, reduce::Reduce, scalar::Mul},
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
            LinalgError::IncompatibleVectors(a, b) =>
                write!(f, "Lengths of vectors are incompatible: {} vs {}.", a, b),
            LinalgError::IncompatibleMatrices(a, b) =>
                write!(f, "Matrices are of incompatible shapes: {:?} vs {:?}.", a, b),
        }
    }
}

impl<E1, E2> std::error::Error for LinalgError<E1, E2>
where
    E1: std::fmt::Debug + std::fmt::Display,
    E2: std::fmt::Debug + std::fmt::Display,
{}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Inner Products
///////////////////////////////////////////////////////////////////////////////////////////////////
impl_newtype!(InnerProduct<N1, N2>(Reduce<Mul<N1, N2>>));

impl<N1, N2> InnerProduct<N1, N2> {
    pub fn new(n1: N1, n2: N2) -> Self {
        InnerProduct(Reduce(Mul(n1, n2)))
    }
}

impl<N1: std::fmt::Display, N2: std::fmt::Display> std::fmt::Display for InnerProduct<N1, N2> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "\u{27E8}{}, {}\u{27E9}", self.0.0.0, self.0.0.1)
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Outer Products
///////////////////////////////////////////////////////////////////////////////////////////////////
pub trait OuterProductTrait<T>: Buffer
where
    T: Buffer<Field = Self::Field>,
{
    type Output: Buffer;

    fn outer_product<E1, E2>(&self, rhs: &T) -> Result<Self::Output, LinalgError<E1, E2>>;
}

impl<F, S> OuterProductTrait<ArrayBase<S, Ix1>> for ArrayBase<S, Ix1>
where
    F: ScalarBuffer + 'static,
    S: ndarray::Data<Elem = F> + ndarray::RawDataClone,
{
    type Output = ndarray::Array2<F>;

    fn outer_product<E1, E2>(&self, rhs: &ArrayBase<S, Ix1>) -> Result<
        ndarray::Array2<F>,
        LinalgError<E1, E2>
    >
    {
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
    F: ScalarBuffer + 'static,
{
    type Output = ndarray::Array2<F>;

    fn outer_product<E1, E2>(&self, rhs: &Vec<F>) -> Result<
        ndarray::Array2<F>,
        LinalgError<E1, E2>
    > {
        ndarray::aview1(self).outer_product(&ndarray::aview1(rhs))
    }
}

pub struct OuterProduct<N1, N2>(pub N1, pub N2);

impl<N1, N2> Node for OuterProduct<N1, N2> {}

impl<T, N1, N2> Contains<T> for OuterProduct<N1, N2>
where
    T: Identifier,
    N1: Contains<T>,
    N2: Contains<T>,
{
    fn contains(&self, target: T) -> bool {
        self.0.contains(target) || self.1.contains(target)
    }
}

impl<S, N1, N2> Function<S> for OuterProduct<N1, N2>
where
    S: State,
    N1: Function<S>,
    N2: Function<S>,

    N1::Codomain: OuterProductTrait<N2::Codomain>,
    N2::Codomain: Buffer<Field = FieldOf<N1::Codomain>>,
{
    type Codomain = <N1::Codomain as OuterProductTrait<N2::Codomain>>::Output;
    type Error = LinalgError<N1::Error, N2::Error>;

    fn evaluate(&self, state: &S) -> Result<Self::Codomain, Self::Error> {
        let x = self.0.evaluate(state).map_err(|e| LinalgError::Inner1(e))?;
        let y = self.1.evaluate(state).map_err(|e| LinalgError::Inner2(e))?;

        x.outer_product(&y)
    }
}

impl<S, T, N1, N2> Differentiable<S, T> for OuterProduct<N1, N2>
where
    S: State,
    T: Identifier,
    N1: Differentiable<S, T>,
    N2: Differentiable<S, T>,

    N1::Codomain: OuterProductTrait<N2::Codomain>,
    N2::Codomain: Buffer<Field = FieldOf<N1::Codomain>>,

    N1::Jacobian: OuterProductTrait<N2::Codomain>,
    N1::Codomain: OuterProductTrait<N2::Jacobian>,

    <N1::Jacobian as OuterProductTrait<N2::Codomain>>::Output:
        std::ops::Add<<N1::Codomain as OuterProductTrait<N2::Jacobian>>::Output>,

    AddOut<
        <N1::Jacobian as OuterProductTrait<N2::Codomain>>::Output,
        <N1::Codomain as OuterProductTrait<N2::Jacobian>>::Output
    >: Buffer<Field = FieldOf<Self::Codomain>>,
{
    type Jacobian = AddOut<
        <N1::Jacobian as OuterProductTrait<N2::Codomain>>::Output,
        <N1::Codomain as OuterProductTrait<N2::Jacobian>>::Output
    >;

    fn grad(&self, state: &S, target: T) -> Result<Self::Jacobian, Self::Error> {
        let dual_x = self.0.dual(state, target).map_err(|e| LinalgError::Inner1(e))?;
        let dual_y = self.1.dual(state, target).map_err(|e| LinalgError::Inner2(e))?;

        dual_x
            .adjoint
            .outer_product(&dual_y.value)
            .and_then(|x| {
                dual_x
                    .value
                    .outer_product(&dual_y.adjoint)
                    .map(|y| x + y)
            })
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

impl<F, S> MatMulTrait<ArrayBase<S, Ix2>> for ArrayBase<S, Ix2>
where
    F: ScalarBuffer + 'static,
    S: ndarray::Data<Elem = F> + ndarray::RawDataClone,
{
    type Output = ndarray::Array2<F>;

    fn mat_mul<E1, E2>(&self, rhs: &ArrayBase<S, Ix2>) -> Result<ndarray::Array2<F>, LinalgError<E1, E2>>
    {
        if self.ncols() == rhs.nrows() {
            Ok(self.dot(rhs))
        } else {
            Err(LinalgError::IncompatibleMatrices(self.dim(), rhs.dim()))
        }
    }
}

pub struct MatMul<N1, N2>(pub N1, pub N2);

impl<N1, N2> Node for MatMul<N1, N2> {}

impl<T, N1, N2> Contains<T> for MatMul<N1, N2>
where
    T: Identifier,
    N1: Contains<T>,
    N2: Contains<T>,
{
    fn contains(&self, target: T) -> bool {
        self.0.contains(target) || self.1.contains(target)
    }
}

impl<S, N1, N2> Function<S> for MatMul<N1, N2>
where
    S: State,
    N1: Function<S>,
    N2: Function<S>,

    N1::Codomain: MatMulTrait<N2::Codomain>,
    N2::Codomain: Buffer<Field = <N1::Codomain as Buffer>::Field>,
{
    type Codomain = <N1::Codomain as MatMulTrait<N2::Codomain>>::Output;
    type Error = LinalgError<N1::Error, N2::Error>;

    fn evaluate(&self, state: &S) -> Result<Self::Codomain, Self::Error> {
        let x = self.0.evaluate(state).map_err(|e| LinalgError::Inner1(e))?;
        let y = self.1.evaluate(state).map_err(|e| LinalgError::Inner2(e))?;

        x.mat_mul(&y)
    }
}

impl<S, T, N1, N2> Differentiable<S, T> for MatMul<N1, N2>
where
    S: State,
    T: Identifier,
    N1: Differentiable<S, T>,
    N2: Differentiable<S, T>,

    N1::Codomain: MatMulTrait<N2::Codomain>,
    N2::Codomain: Buffer<Field = FieldOf<N1::Codomain>>,

    N1::Jacobian: MatMulTrait<N2::Codomain>,
    N1::Codomain: MatMulTrait<N2::Jacobian>,

    <N1::Jacobian as MatMulTrait<N2::Codomain>>::Output:
        std::ops::Add<<N1::Codomain as MatMulTrait<N2::Jacobian>>::Output>,

    AddOut<
        <N1::Jacobian as MatMulTrait<N2::Codomain>>::Output,
        <N1::Codomain as MatMulTrait<N2::Jacobian>>::Output
    >: Buffer<Field = FieldOf<Self::Codomain>>,
{
    type Jacobian = AddOut<
        <N1::Jacobian as MatMulTrait<N2::Codomain>>::Output,
        <N1::Codomain as MatMulTrait<N2::Jacobian>>::Output
    >;

    fn grad(&self, state: &S, target: T) -> Result<Self::Jacobian, Self::Error> {
        let dual_x = self.0.dual(state, target).map_err(|e| LinalgError::Inner1(e))?;
        let dual_y = self.1.dual(state, target).map_err(|e| LinalgError::Inner2(e))?;

        dual_x
            .adjoint
            .mat_mul(&dual_y.value)
            .and_then(|x| {
                dual_x
                    .value
                    .mat_mul(&dual_y.adjoint)
                    .map(|y| x + y)
            })
    }
}
