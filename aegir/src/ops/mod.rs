//! Module for concrete operator implementations.
macro_rules! impl_unary {
    ($(#[$attr:meta])* $name:ident<$F:ident: $field_type:path>, |$x:ident| $f:block, |$self:ident| $e:block) => {
        $(#[$attr])*
        #[derive(Clone, Copy, Debug, PartialEq, Contains)]
        pub struct $name<N>(#[op] pub N);

        impl<N: crate::Node> crate::Node for $name<N> {}

        impl<D, N, $F> crate::Function<D> for $name<N>
        where
            D: crate::Database,
            N: crate::Function<D>,
            $F: crate::buffers::Scalar + $field_type,

            N::Value: crate::buffers::Buffer<Field = $F>,
        {
            type Error = N::Error;
            type Value = N::Value;

            fn evaluate<DR: AsRef<D>>(&self, db: DR) -> Result<Self::Value, Self::Error> {
                use crate::buffers::Buffer;

                self.0.evaluate(db).map(|mut buf| { buf.mutate(|$x| $f); buf })
            }

            fn evaluate_spec<DR: AsRef<D>>(&self, db: DR) -> Result<crate::buffers::Spec<Self::Value>, Self::Error> {
                use crate::buffers::{Buffer, Spec::*};

                Ok(match self.0.evaluate_spec(db)? {
                    Full(sh, $x) => Full(sh, $f),
                    spec => Raw({
                        let mut buf = spec.unwrap();

                        buf.mutate(|$x| $f);

                        buf
                    }),
                })
            }

            fn evaluate_shape<DR: AsRef<D>>(&self, db: DR) -> Result<crate::buffers::shapes::ShapeOf<Self::Value>, Self::Error> {
                self.0.evaluate_shape(db)
            }
        }

        impl<N: crate::fmt::ToExpr> crate::fmt::ToExpr for $name<N> {
            fn to_expr(&$self) -> crate::fmt::Expr { $e }
        }

        impl<N: crate::fmt::ToExpr> std::fmt::Display for $name<N> {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                use crate::fmt::ToExpr;

                self.to_expr().fmt(f)
            }
        }
    };
}

mod arithmetic;
pub use self::arithmetic::*;

mod linalg;
pub use self::linalg::*;

mod logarithmic;
pub use self::logarithmic::*;

mod trig;
pub use self::trig::*;

mod sigmoid;
pub use self::sigmoid::*;

mod special;
pub use self::special::*;
