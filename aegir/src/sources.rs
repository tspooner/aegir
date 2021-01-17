use crate::{
    Identifier, Database, Get, Node, Contains,
    Function, Differentiable, Compile, Prune,
    buffer::{Buffer, Field, OwnedBuffer, OwnedOf, FieldOf},
};

#[derive(Copy, Clone, Debug)]
pub struct VariableError<ID>(ID);

impl<ID: std::fmt::Debug> std::fmt::Display for VariableError<ID> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Failed to get {:?} from data source.", self.0)
    }
}

impl<ID: std::fmt::Debug> std::error::Error for VariableError<ID> {}

#[derive(Copy, Clone, Debug)]
pub struct Variable<I>(pub I);

impl<I> Node for Variable<I> {}

impl<T, I> Contains<T> for Variable<I>
where
    T: Identifier,
    I: Identifier + std::cmp::PartialEq<T>,
{
    fn contains(&self, target: T) -> bool { self.0 == target }
}

impl<D, I> Function<D> for Variable<I>
where
    D: Get<I>,
    I: Identifier,

    D::Output: Buffer,
{
    type Codomain = OwnedOf<D::Output>;
    type Error = VariableError<I>;

    fn evaluate(&self, db: &D) -> Result<Self::Codomain, Self::Error> {
        db
            .get(self.0)
            .map(|v| v.to_owned())
            .ok_or_else(|| VariableError(self.0))
    }
}

impl<D, T, I> Differentiable<D, T> for Variable<I>
where
    D: Get<T> + Get<I>,
    T: Identifier,
    I: Identifier + std::cmp::PartialEq<T>,

    <D as Get<T>>::Output: Buffer,
    <D as Get<I>>::Output: Buffer<Field = FieldOf<<D as Get<T>>::Output>>,

    FieldOf<<D as Get<T>>::Output>: num_traits::One + num_traits::Zero,
{
    type Jacobian = OwnedOf<<D as Get<I>>::Output>;

    fn grad(&self, db: &D, target: T) -> Result<Self::Jacobian, Self::Error> {
        if self.contains(target) {
            db
                .get(self.0)
                .map(|a| a.to_ones())
                .ok_or_else(|| VariableError(self.0))
        } else {
            db
                .get(self.0)
                .map(|a| a.to_zeroes())
                .ok_or_else(|| VariableError(self.0))
        }
    }
}

impl<T, I> Compile<T> for Variable<I>
where
    T: Identifier,
    I: Identifier + std::cmp::PartialEq<T>,
{
    type CompiledJacobian = GradOf<I>;
    type Error = crate::NoError;

    fn compile_grad(&self, target: T) -> Result<Self::CompiledJacobian, Self::Error> {
        if self.contains(target) {
            Ok(GradOf(GradValue::One, self.0))
        } else {
            Ok(GradOf(GradValue::Zero, self.0))
        }
    }
}

impl<I: std::fmt::Display> std::fmt::Display for Variable<I> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.0, f)
    }
}

impl<I> From<I> for Variable<I> {
    fn from(selector: I) -> Variable<I> { Variable(selector) }
}

#[derive(Copy, Clone, Debug)]
pub enum GradValue { Zero, One }

impl GradValue {
    pub(self) fn convert<B>(&self, buffer: &B) -> B::Owned
    where
        B: Buffer,
        FieldOf<B>: num_traits::Zero + num_traits::One,
    {
        match self {
            GradValue::Zero => buffer.to_zeroes(),
            GradValue::One => buffer.to_ones(),
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct GradOf<I>(pub GradValue, pub I);

impl<I> Node for GradOf<I> {}

impl<T, I> Contains<T> for GradOf<I>
where
    T: Identifier,
    I: Identifier + std::cmp::PartialEq<T>,
{
    fn contains(&self, target: T) -> bool { self.1 == target }
}

impl<D, I> Function<D> for GradOf<I>
where
    D: Get<I>,
    I: Identifier,

    D::Output: Buffer,

    FieldOf<D::Output>: num_traits::Zero + num_traits::One,
{
    type Codomain = OwnedOf<D::Output>;
    type Error = VariableError<I>;

    fn evaluate(&self, db: &D) -> Result<Self::Codomain, Self::Error> {
        db
            .get(self.1)
            .map(|buffer| self.0.convert(buffer))
            .ok_or_else(|| VariableError(self.1))
    }
}

impl<D, T, F, I> Differentiable<D, T> for GradOf<I>
where
    D: Get<I>,
    T: Identifier,
    F: Field + num_traits::Zero + num_traits::One,
    I: Identifier + std::cmp::PartialEq<T>,

    D::Output: Buffer<Field = F>,
{
    type Jacobian = OwnedOf<<D as Get<I>>::Output>;

    fn grad(&self, db: &D, _: T) -> Result<Self::Jacobian, Self::Error> {
        db
            .get(self.1)
            .map(|v| v.to_zeroes())
            .ok_or_else(|| VariableError(self.1))
    }
}

impl<T, I> Compile<T> for GradOf<I>
where
    T: Identifier,
    I: Identifier + std::cmp::PartialEq<T>,
{
    type CompiledJacobian = Self;
    type Error = crate::NoError;

    fn compile_grad(&self, _: T) -> Result<Self::CompiledJacobian, Self::Error> {
        Ok(GradOf(GradValue::Zero, self.1))
    }
}

impl<I> Prune for GradOf<I>
where
    I: Identifier,
{
    fn prune(self) -> Option<Self> {
        match self.0 {
            GradValue::Zero => None,
            GradValue::One => Some(self),
        }
    }
}

impl<I: std::fmt::Display> std::fmt::Display for GradOf<I> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.0 {
            GradValue::Zero => write!(f, "0"),
            GradValue::One => write!(f, "\u{2202}{}", self.1),
        }
    }
}

#[derive(Copy, Clone, Debug)]
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
    B: OwnedBuffer,
{
    type Codomain = OwnedOf<B>;
    type Error = VariableError<()>;

    fn evaluate(&self, _: &D) -> Result<Self::Codomain, Self::Error> {
        Ok(self.0.clone())
    }
}

impl<D, T, B> Differentiable<D, T> for Constant<B>
where
    D: Database,
    T: Identifier,
    B: OwnedBuffer,

    FieldOf<B>: num_traits::Zero,
{
    type Jacobian = B;

    fn grad(&self, _: &D, _: T) -> Result<Self::Jacobian, Self::Error> {
        Ok(self.0.to_zeroes())
    }
}

impl<T, B> Compile<T> for Constant<B>
where
    T: Identifier,
    B: OwnedBuffer,

    FieldOf<B>: num_traits::Zero,
{
    type CompiledJacobian = Constant<B>;
    type Error = crate::NoError;

    fn compile_grad(&self, _: T) -> Result<Self::CompiledJacobian, Self::Error> {
        Ok(Constant(self.0.to_zeroes()))
    }
}

impl<B: std::fmt::Display> std::fmt::Display for Constant<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}
