//! Module for concrete operator implementations.
macro_rules! impl_unary {
    ($(#[$attr:meta])* $name:ident<$F:ident: $field_type:path>, |$x:ident| $f:block, |$self:ident| $e:block) => {
        $(#[$attr])*
        #[derive(Clone, Copy, Debug, PartialEq, Contains)]
        pub struct $name<N>(#[op] pub N);

        impl<N: crate::Node> crate::Node for $name<N> {}

        impl<C, N, $F> crate::Function<C> for $name<N>
        where
            C: crate::Context,
            N: crate::Function<C>,
            $F: crate::buffers::Scalar + $field_type,

            N::Value: crate::buffers::Buffer<Field = $F>,
        {
            type Error = N::Error;
            type Value = N::Value;

            fn evaluate<CR: AsMut<C>>(&self, ctx: CR) -> Result<Self::Value, Self::Error> {
                use crate::buffers::Buffer;

                self.0.evaluate(ctx).map(|mut buf| { buf.mutate(|$x| $f); buf })
            }

            fn evaluate_spec<CR: AsMut<C>>(&self, ctx: CR) -> Result<crate::buffers::Spec<Self::Value>, Self::Error> {
                use crate::buffers::{Buffer, Spec::*};

                Ok(match self.0.evaluate_spec(ctx)? {
                    Full(sh, $x) => Full(sh, $f),
                    spec => Raw({
                        let mut buf = spec.unwrap();

                        buf.mutate(|$x| $f);

                        buf
                    }),
                })
            }

            fn evaluate_shape<CR: AsMut<C>>(&self, ctx: CR) -> Result<crate::buffers::shapes::ShapeOf<Self::Value>, Self::Error> {
                self.0.evaluate_shape(ctx)
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

mod sigmoid;
pub use self::sigmoid::*;

pub mod trig;
pub mod special;
