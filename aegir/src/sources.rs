use crate::{
    Function, Differentiable, Node, Identifier, Get,
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

impl<I: Identifier, S: Get<I>> Function<S> for Variable<I>
where
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

impl<I, T, S> Differentiable<T, S> for Variable<I>
where
    I: Identifier + std::cmp::PartialEq<T>,
    T: Identifier,
    S: Get<I> + Get<T>,

    <S as Get<T>>::Output: Buffer,
    <S as Get<I>>::Output: Buffer<Field = FieldOf<<S as Get<T>>::Output>>,
{
    type Jacobian = OwnedOf<<S as Get<I>>::Output>;

    fn grad(&self, target: T, state: &S) -> Result<Self::Jacobian, Self::Error> {
        if self.0 == target {
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

impl<B: OwnedBuffer, S> Function<S> for Constant<B> {
    type Codomain = OwnedOf<B>;
    type Error = VariableError<()>;

    fn evaluate(&self, _: &S) -> Result<Self::Codomain, Self::Error> {
        Ok(self.0.clone())
    }
}

impl<B, T, S> Differentiable<T, S> for Constant<B>
where
    B: OwnedBuffer + Get<T>,
    T: Identifier,
{
    type Jacobian = B;

    fn grad(&self, _: T, _: &S) -> Result<Self::Jacobian, Self::Error> {
        Ok(self.0.to_zeroes())
    }
}

impl<B: std::fmt::Display> std::fmt::Display for Constant<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}
