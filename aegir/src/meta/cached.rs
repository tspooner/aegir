use crate::{
    Node,
    Write,
    Contains,
    Function,
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
