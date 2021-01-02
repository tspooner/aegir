pub struct MergedState<I, M> {
    pub input: I,
    pub memory: M,
}

impl<T, I, M> crate::Get<T> for MergedState<I, M>
where
    T: crate::Identifier,
    I: crate::Get<T>,
    M: crate::Get<T, Output = I::Output>,
{
    type Output = I::Output;

    fn get(&self, target: T) -> Result<&I::Output, crate::GetError<T>> {
        self.input.get(target).or_else(|_| self.memory.get(target))
    }
}

pub struct Module<F, M> {
    pub op: F,
    pub memory: M,
}

impl<F, M> crate::Node for Module<F, M> {}

impl<S, F, M> crate::Function<S> for Module<F, M>
where
    F: crate::Function<MergedState<S, M>> + for<'a, 'b> crate::Function<
        MergedState<&'a S, &'b M>,
        Codomain = <F as crate::Function<MergedState<S, M>>>::Codomain,
        Error = <F as crate::Function<MergedState<S, M>>>::Error,
    >,
{
    type Codomain = <F as crate::Function<MergedState<S, M>>>::Codomain;
    type Error = <F as crate::Function<MergedState<S, M>>>::Error;

    fn evaluate(&self, state: &S) -> Result<Self::Codomain, Self::Error> {
        self.op.evaluate(&MergedState {
            input: state,
            memory: &self.memory,
        })
    }
}
