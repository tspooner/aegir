use crate::{
    Identifier, State, Get, Node, Contains, Function, Differentiable,
    buffer::{Buffer, OwnedBuffer, OwnedOf, FieldOf},
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

impl<S, I> Function<S> for Variable<I>
where
    S: Get<I>,
    I: Identifier,

    S::Output: Buffer,
{
    type Codomain = OwnedOf<S::Output>;
    type Error = VariableError<I>;

    fn evaluate(&self, state: &S) -> Result<Self::Codomain, Self::Error> {
        state
            .get(self.0)
            .map(|v| v.to_owned())
            .ok_or_else(|| VariableError(self.0))
    }
}

impl<S, T, I> Differentiable<S, T> for Variable<I>
where
    S: Get<T> + Get<I>,
    T: Identifier,
    I: Identifier + std::cmp::PartialEq<T>,

    <S as Get<T>>::Output: Buffer,
    <S as Get<I>>::Output: Buffer<Field = FieldOf<<S as Get<T>>::Output>>,
{
    type Jacobian = OwnedOf<<S as Get<I>>::Output>;

    fn grad(&self, state: &S, target: T) -> Result<Self::Jacobian, Self::Error> {
        if self.contains(target) {
            state
                .get(self.0)
                .map(|a| a.to_ones())
                .ok_or_else(|| VariableError(self.0))
        } else {
            state
                .get(self.0)
                .map(|a| a.to_zeroes())
                .ok_or_else(|| VariableError(self.0))
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
pub struct Constant<B>(pub B);

impl<B> Node for Constant<B> {}

impl<T, B> Contains<T> for Constant<B>
where
    T: Identifier,
{
    fn contains(&self, _: T) -> bool { false }
}

impl<S, B> Function<S> for Constant<B>
where
    S: State,
    B: OwnedBuffer,
{
    type Codomain = OwnedOf<B>;
    type Error = VariableError<()>;

    fn evaluate(&self, _: &S) -> Result<Self::Codomain, Self::Error> {
        Ok(self.0.clone())
    }
}

impl<S, T, B> Differentiable<S, T> for Constant<B>
where
    S: State,
    T: Identifier,
    B: OwnedBuffer + Get<T>,
{
    type Jacobian = B;

    fn grad(&self, _: &S, _: T) -> Result<Self::Jacobian, Self::Error> {
        Ok(self.0.to_zeroes())
    }
}

impl<B: std::fmt::Display> std::fmt::Display for Constant<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}
