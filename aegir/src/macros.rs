macro_rules! new_op {
    ($name:ident<$($tp:ident),+>) => {
        #[derive(Copy, Clone, Debug)]
        pub struct $name<$($tp),+>($(pub $tp),+);

        impl<$($tp),+> crate::Node for $name<$($tp),+> {}
    };
    ($name:ident<$($tp:ident),+>($inner:ty)) => {
        #[derive(Copy, Clone, Debug)]
        pub struct $name<$($tp),+>(pub $inner);

        impl<$($tp),+> crate::Node for $name<$($tp),+> {}
    };
}

macro_rules! impl_newtype {
    ($name:ident<$($tp:ident),+>($inner:ty)) => {
        new_op!($name<$($tp),+>($inner));

        impl<T, $($tp),+> crate::Contains<T> for $name<$($tp),+>
        where
            T: crate::Identifier,
            $inner: crate::Contains<T>,
        {
            fn contains(&self, target: T) -> bool { self.0.contains(target) }
        }

        impl<S, $($tp),+> crate::Function<S> for $name<$($tp),+>
        where
            S: crate::State,
            $inner: crate::Function<S>,
        {
            type Codomain = crate::CodomainOf<$inner, S>;
            type Error = crate::ErrorOf<$inner, S>;

            fn evaluate(&self, state: &S) -> Result<Self::Codomain, Self::Error> {
                self.0.evaluate(state)
            }
        }

        impl<S, T, $($tp),+> crate::Differentiable<S, T> for $name<$($tp),+>
        where
            S: crate::State,
            T: crate::Identifier,
            $inner: crate::Differentiable<S, T>,
        {
            type Jacobian = crate::JacobianOf<$inner, S, T>;

            fn grad(&self, state: &S, target: T) -> Result<Self::Jacobian, Self::Error> {
                self.0.grad(state, target)
            }
        }
    }
}
