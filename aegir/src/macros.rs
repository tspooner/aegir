macro_rules! impl_newtype {
    ($(#[$attr:meta])* $name:ident<$($tp:ident),+>($inner:ty)) => {
        $(#[$attr])*
        #[derive(Copy, Clone, Debug, PartialEq, Node, Contains)]
        pub struct $name<$($tp: PartialEq),+>(#[op] pub $inner);

        impl<D, $($tp),+> crate::Function<D> for $name<$($tp),+>
        where
            D: crate::Database,
            $inner: crate::Function<D>,
            $($tp: PartialEq),+
        {
            type Codomain = crate::CodomainOf<$inner, D>;
            type Error = crate::ErrorOf<$inner, D>;

            fn evaluate(&self, db: &D) -> Result<Self::Codomain, Self::Error> {
                self.0.evaluate(db)
            }
        }

        impl<D, T, $($tp),+> crate::Differentiable<D, T> for $name<$($tp),+>
        where
            D: crate::Database,
            T: crate::Identifier,
            $inner: crate::Differentiable<D, T>,
            $($tp: PartialEq),+
        {
            type Jacobian = crate::JacobianOf<$inner, D, T>;

            fn grad(&self, db: &D, target: T) -> Result<Self::Jacobian, Self::Error> {
                self.0.grad(db, target)
            }

            fn dual(&self, db: &D, target: T) -> Result<crate::Dual<Self::Codomain, Self::Jacobian>, Self::Error> {
                self.0.dual(db, target)
            }
        }
    }
}
