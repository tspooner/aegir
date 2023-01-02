use crate::{
    Node,
    Write,
    Contains,
    Function,
    Differentiable,
    Identifier,
    AegirResult,
};

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Cached<N, T> {
    pub node: N,
    pub ident: T,
}

impl<N: Node, T: Identifier> Node for Cached<N, T> {}

impl<N, T, I> Contains<I> for Cached<N, T>
where
    N: Contains<I>,
    T: Identifier,
    I: Identifier + PartialEq<T>,
{
    fn contains(&self, ident: I) -> bool {
        ident == self.ident || self.node.contains(ident)
    }
}

impl<N, T, C> Function<C> for Cached<N, T>
where
    N: Function<C, Value = C::Buffer>,
    T: Identifier,
    C: Write<T>,
{
    type Error = N::Error;
    type Value = N::Value;

    fn evaluate<CR: AsMut<C>>(&self, mut ctx: CR) -> AegirResult<Self, C> {
        let cached = ctx.as_mut().read(self.ident);

        if let Some(value) = cached {
            return Ok(value);
        }

        let value = self.node.evaluate(ctx.as_mut())?;

        ctx.as_mut().write(self.ident, value.clone());

        Ok(value)
    }
}

impl<N, T> Differentiable<T> for Cached<N, T>
where
    N: Differentiable<T>,
    T: Identifier,
{
    type Adjoint = N::Adjoint;

    fn adjoint(&self, target: T) -> Self::Adjoint {
        // TODO - find a way to propagate the cached terms through the derivative.
        // This will require some kind of search in which find-and-replace all instances of
        // N in Self::Adjoint with Cached<N, T>.
        self.node.adjoint(target)
    }
}
