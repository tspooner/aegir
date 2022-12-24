use crate::{
    buffers::{
        shapes::{self, Shape, Zip, Shaped, ShapeOf},
        Buffer,
        Class,
        IncompatibleShapes,
        Scalar,
        Spec,
        ZipMap,
    },
    fmt::{ToExpr, Expr, PreWrap},
    ops::SafeXlnX,
    BinaryError,
    Contains,
    Context,
    Differentiable,
    Function,
    Identifier,
    Node,
};
use std::fmt;

type Error<C, L, R> = BinaryError<
    <L as Function<C>>::Error,
    <R as Function<C>>::Error,
    IncompatibleShapes<ShapeOf<<L as Function<C>>::Value>, ShapeOf<<R as Function<C>>::Value>>,
>;

/// Computes the power of a [Buffer] to a [Field].
///
/// # Examples
/// ## x^2
/// ```
/// # #[macro_use] extern crate aegir;
/// # use aegir::{Identifier, Differentiable, Dual, buffers::Buffer, ops::Power, ids::X};
/// ctx!(Ctx { x: X });
///
/// let f = Power(X.into_var(), 2.0f64.into_constant());
///
/// assert_eq!(f.evaluate_dual(X, &Ctx { x: -1.0 }).unwrap(), dual!(1.0, -2.0));
/// assert_eq!(f.evaluate_dual(X, &Ctx { x: 0.0 }).unwrap(), dual!(0.0, 0.0));
/// assert_eq!(f.evaluate_dual(X, &Ctx { x: 1.0 }).unwrap(), dual!(1.0, 2.0));
/// assert_eq!(f.evaluate_dual(X, &Ctx { x: 2.0 }).unwrap(), dual!(4.0, 4.0));
/// ```
///
/// ## x^y
/// ```
/// # #[macro_use] extern crate aegir;
/// # use aegir::{Identifier, Differentiable, Dual, buffers::Buffer, ops::Power, ids::{X, Y}};
/// # use aegir::{Function};
/// ctx!(Ctx { x: X, y: Y });
///
/// let f = Power(X.into_var(), Y.into_var());
///
/// assert!((
///     f.evaluate(&Ctx { x: 2.0, y: 1.5, }).unwrap() - 2.0f64.powf(1.5)
/// ) < 1e-5);
/// assert!((
///     f.evaluate_adjoint(X, &Ctx { x: 2.0, y: 1.5, }).unwrap() - 1.5 * 2.0f64.powf(0.5)
/// ) < 1e-5);
/// assert!((
///     f.evaluate_adjoint(Y, &Ctx { x: 2.0, y: 1.5, }).unwrap() - 2.0f64.powf(1.5) * 2.0f64.ln()
/// ) < 1e-5);
/// ```
///
/// # Derivation
/// We can derive the gradient computation as follows.
///
/// First, let
///      z(x) := f(x) ^ g(x).
/// Then, to get the derivative we first take logarithms,
///      ln z(x) = g(x) * ln f(x),
/// such that
///      z'(x) / z(x) = g'(x) * ln f(x) + f'(x) * g(x) / f(x).
/// It then follows that
///      z'(x) = z(x) * [g'(x) * ln f(x) + f'(x) * g(x) / f(x)]
/// as desired.
///
/// For simplicity, let
///      l(x) := z(x) * g'(x) * ln f(x),
/// and
///      r(x) := z(x) * f'(x) * g(x) / f(x).
/// Then
///      z'(x) = l(x) + r(x).
#[derive(Copy, Clone, Debug, PartialEq, Contains)]
pub struct Power<N, E>(#[op] pub N, #[op] pub E);

impl<N: Node, E: Node> Node for Power<N, E> {}

impl<F, C, N, E> Function<C> for Power<N, E>
where
    F: Scalar + num_traits::Pow<F, Output = F>,
    C: Context,

    N: Function<C, Value = F>,
    E: Function<C, Value = F>,
{
    type Error = BinaryError<N::Error, E::Error, crate::NoError>;
    type Value = N::Value;

    fn evaluate<CR: AsRef<C>>(&self, ctx: CR) -> Result<Self::Value, Self::Error> {
        self.evaluate_spec(ctx).map(|state| state.unwrap())
    }

    fn evaluate_spec<CR: AsRef<C>>(&self, ctx: CR) -> Result<Spec<Self::Value>, Self::Error> {
        use Spec::*;

        let zero = F::zero();
        let one = F::one();

        let base = self.0.evaluate_spec(&ctx).map_err(BinaryError::Left)?;
        let exponent = self.1.evaluate_spec(ctx).map_err(BinaryError::Right)?;

        match (base, exponent) {
            (b, Full(_, fx)) if fx == zero => Ok(Spec::ones(b.shape())),

            (Full(sx, fx), _) if fx == one => Ok(Full(sx, fx)),
            (Full(sx, fx), _) if fx == zero => Ok(Full(sx, fx)),

            (b, e) => {
                let e = e.unwrap();
                let mut b = b.unwrap();

                b.mutate(|x| x.pow(e));

                Ok(Raw(b))
            },
        }
    }

    fn evaluate_shape<CR: AsRef<C>>(&self, _: CR) -> Result<shapes::S0, Self::Error> {
        Ok(shapes::S0)
    }
}

impl<T, N, E> Differentiable<T> for Power<N, E>
where
    T: Identifier,
    N: Differentiable<T> + Clone,
    E: Differentiable<T> + Clone,
{
    type Adjoint = Mul<
        Power<N, crate::ops::SubOne<E>>,
        Add<Mul<E, N::Adjoint>, Mul<SafeXlnX<N>, E::Adjoint>>,
    >;

    fn adjoint(&self, target: T) -> Self::Adjoint {
        let l = Power(self.0.clone(), crate::ops::SubOne(self.1.clone()));
        let r_l = Mul(self.1.clone(), self.0.adjoint(target));
        let r_r = Mul(SafeXlnX(self.0.clone()), self.1.adjoint(target));

        l.mul(r_l.add(r_r))
    }
}

impl<N: ToExpr, E: ToExpr> ToExpr for Power<N, E> {
    fn to_expr(&self) -> Expr {
        use Expr::*;

        match (self.0.to_expr(), self.1.to_expr()) {
            (Zero, _) => Zero,
            (_, Zero) => One,
            (One, _) => One,
            (l, One) => l,
            (Text(l), Text(r)) => Text(PreWrap {
                text: format!("{}^{}", l.to_safe_string('(', ')'), r.to_safe_string('(', ')')),
                needs_wrap: false,
            })
        }
    }
}

impl<X: ToExpr, E: ToExpr> fmt::Display for Power<X, E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.to_expr().fmt(f)
    }
}

/// Computes the (elementwise) addition of two compatible [Buffer] types.
///
/// # Examples
/// ## x + y
/// ```
/// # #[macro_use] extern crate aegir;
/// # use aegir::{Identifier, Differentiable, Dual, ops::Add, ids::{X, Y}};
/// ctx!(Ctx { x: X, y: Y });
///
/// let f = Add(X.into_var(), Y.into_var());
///
/// assert_eq!(f.evaluate_dual(X, &Ctx { x: 1.0, y: 2.0, }).unwrap(), dual!(3.0, 1.0));
/// assert_eq!(f.evaluate_dual(Y, &Ctx { x: 1.0, y: 2.0, }).unwrap(), dual!(3.0, 1.0));
/// ```
///
/// ## x + y^2
/// ```
/// # #[macro_use] extern crate aegir;
/// # use aegir::{Identifier, Node, Differentiable, Dual, buffers::Buffer, ops::Add, ids::{X, Y}};
/// ctx!(Ctx { x: X, y: Y });
///
/// let f = Add(X.into_var(), Y.into_var().pow(2.0f64.into_constant()));
///
/// assert_eq!(f.evaluate_dual(X, &Ctx { x: 1.0, y: 2.0, }).unwrap(), dual!(5.0, 1.0));
/// assert_eq!(f.evaluate_dual(Y, &Ctx { x: 1.0, y: 2.0, }).unwrap(), dual!(5.0, 4.0));
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Contains)]
pub struct Add<L, R>(#[op] pub L, #[op] pub R);

impl<L: Node, R: Node> Node for Add<L, R> {}

impl<C, F, L, LV, R, RV, OV> Function<C> for Add<L, R>
where
    C: Context,
    F: Scalar,

    L: Function<C, Value = LV>,
    LV: Buffer<Field = F> + ZipMap<RV, Output<F> = OV>,

    R: Function<C, Value = RV>,
    RV: Buffer<Field = F>,

    OV: Buffer<Field = F>,

    LV::Shape: Zip<RV::Shape, Shape = OV::Shape>,
{
    type Error = Error<C, L, R>;
    type Value = OV;

    fn evaluate<CR: AsRef<C>>(&self, ctx: CR) -> Result<Self::Value, Self::Error> {
        self.evaluate_spec(ctx).map(|state| state.unwrap())
    }

    fn evaluate_spec<CR: AsRef<C>>(&self, ctx: CR) -> Result<Spec<Self::Value>, Self::Error> {
        use Spec::*;

        let x = self.0.evaluate_spec(&ctx).map_err(BinaryError::Left)?;
        let y = self.1.evaluate_spec(ctx).map_err(BinaryError::Right)?;

        match (x, y) {
            (Full(sx, fx), Full(sy, fy)) if (fx + fy).is_zero() => {
                sx.zip(sy).map(Spec::zeroes).map_err(BinaryError::Output)
            },

            (Full(sx, fx), Full(sy, fy)) if (fx + fy).is_one() => {
                sx.zip(sy).map(Spec::ones).map_err(BinaryError::Output)
            },

            (Full(sx, fx), Full(sy, fy)) => {
                let shape = sx.zip(sy).map_err(BinaryError::Output)?;
                let buffer = <OV::Class as Class<OV::Shape>>::full(shape, fx + fy);

                Ok(Raw(buffer))
            },

            (x, y) => {
                let x = x.unwrap();
                let y = y.unwrap();

                Ok(Raw(x.zip_map(y, |xi, yi| xi + yi).unwrap()))
            },
        }
    }

    fn evaluate_shape<CR: AsRef<C>>(&self, ctx: CR) -> Result<ShapeOf<Self::Value>, Self::Error> {
        let l_shape = self.0.evaluate_shape(&ctx).map_err(BinaryError::Left)?;
        let r_shape = self.1.evaluate_shape(ctx).map_err(BinaryError::Right)?;

        l_shape.zip(r_shape).map_err(BinaryError::Output)
    }
}

impl<T, L, R> Differentiable<T> for Add<L, R>
where
    T: Identifier,
    L: Differentiable<T>,
    R: Differentiable<T>,
{
    type Adjoint = Add<L::Adjoint, R::Adjoint>;

    fn adjoint(&self, target: T) -> Self::Adjoint {
        let x = self.0.adjoint(target);
        let y = self.1.adjoint(target);

        Add(x, y)
    }
}

impl<L: ToExpr, R: ToExpr> ToExpr for Add<L, R> {
    fn to_expr(&self) -> Expr {
        use Expr::*;

        match (self.0.to_expr(), self.1.to_expr()) {
            (Zero, Zero) => Zero,
            (Zero, One) | (One, Zero) => One,

            (One, One) => Text(PreWrap {
                text: "2".to_string(),
                needs_wrap: false,
            }),

            (Text(l), Zero) => Text(l),
            (Zero, Text(r)) => Text(r),

            (Text(l), One) => Text(PreWrap {
                text: format!("{} + 1", l.to_safe_string('(', ')')),
                needs_wrap: true,
            }),
            (One, Text(r)) => Text(PreWrap {
                text: format!("1 + {}", r.to_safe_string('(', ')')),
                needs_wrap: true,
            }),
            (Text(l), Text(r)) => Text(PreWrap {
                text: format!("{} + {}", l.to_safe_string('(', ')'), r.to_safe_string('(', ')')),
                needs_wrap: true,
            })
        }
    }
}

impl<L: ToExpr, R: ToExpr> std::fmt::Display for Add<L, R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.to_expr().fmt(f)
    }
}

/// Computes the (elementwise) subtraction of two compatible [Buffer] types.
///
/// # Examples
/// ## x - y
/// ```
/// # #[macro_use] extern crate aegir;
/// # use aegir::{Identifier, Differentiable, Dual, buffers::Buffer, ops::Sub, ids::{X, Y}};
/// ctx!(Ctx { x: X, y: Y });
///
/// let f = Sub(X.into_var(), Y.into_var());
///
/// assert_eq!(f.evaluate_dual(X, &Ctx { x: 1.0, y: 2.0, }).unwrap(), dual!(-1.0, 1.0));
/// assert_eq!(f.evaluate_dual(Y, &Ctx { x: 1.0, y: 2.0, }).unwrap(), dual!(-1.0, -1.0));
/// ```
///
/// ## x - y^2
/// ```
/// # #[macro_use] extern crate aegir;
/// # use aegir::{Identifier, Node, Differentiable, Dual, buffers::Buffer, ops::Sub, ids::{X, Y}};
/// ctx!(Ctx { x: X, y: Y });
///
/// let f = Sub(X.into_var(), Y.into_var().pow(2.0f64.into_constant()));
///
/// assert_eq!(f.evaluate_dual(X, &Ctx { x: 1.0, y: 2.0, }).unwrap(), dual!(-3.0, 1.0));
/// assert_eq!(f.evaluate_dual(Y, &Ctx { x: 1.0, y: 2.0, }).unwrap(), dual!(-3.0, -4.0));
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Contains)]
pub struct Sub<L, R>(#[op] pub L, #[op] pub R);

impl<L: Node, R: Node> Node for Sub<L, R> {}

impl<C, F, L, LV, R, RV, OV> Function<C> for Sub<L, R>
where
    C: Context,
    F: Scalar,

    L: Function<C, Value = LV>,
    LV: Buffer<Field = F> + ZipMap<RV, Output<F> = OV>,

    R: Function<C, Value = RV>,
    RV: Buffer<Field = F> + ZipMap<LV, Output<F> = OV>,

    OV: Buffer<Field = F>,

    LV::Shape: Zip<RV::Shape, Shape = OV::Shape>,
{
    type Error = Error<C, L, R>;
    type Value = OV;

    fn evaluate<CR: AsRef<C>>(&self, ctx: CR) -> Result<Self::Value, Self::Error> {
        self.evaluate_spec(ctx).map(|state| state.unwrap())
    }

    fn evaluate_spec<CR: AsRef<C>>(&self, ctx: CR) -> Result<Spec<Self::Value>, Self::Error> {
        use Spec::*;

        let x = self.0.evaluate_spec(&ctx).map_err(BinaryError::Left)?;
        let y = self.1.evaluate_spec(ctx).map_err(BinaryError::Right)?;

        match (x, y) {
            (Full(sx, fx), Full(sy, fy)) if (fx - fy).is_zero() => {
                sx.zip(sy).map(Spec::zeroes).map_err(BinaryError::Output)
            },

            (Full(sx, fx), Full(sy, fy)) if (fx - fy).is_one() => {
                sx.zip(sy).map(Spec::ones).map_err(BinaryError::Output)
            },

            (x, y) => {
                let x = x.unwrap();
                let y = y.unwrap();

                Ok(Raw(x.zip_map(y, |xi, yi| xi - yi).unwrap()))
            },
        }
    }

    fn evaluate_shape<CR: AsRef<C>>(&self, ctx: CR) -> Result<ShapeOf<Self::Value>, Self::Error> {
        let l_shape = self.0.evaluate_shape(&ctx).map_err(BinaryError::Left)?;
        let r_shape = self.1.evaluate_shape(ctx).map_err(BinaryError::Right)?;

        l_shape.zip(r_shape).map_err(BinaryError::Output)
    }
}

impl<T, L, R> Differentiable<T> for Sub<L, R>
where
    T: Identifier,
    L: Differentiable<T>,
    R: Differentiable<T>,
{
    type Adjoint = Sub<L::Adjoint, R::Adjoint>;

    fn adjoint(&self, target: T) -> Self::Adjoint {
        let x = self.0.adjoint(target);
        let y = self.1.adjoint(target);

        Sub(x, y)
    }
}

impl<L: ToExpr, R: ToExpr> ToExpr for Sub<L, R> {
    fn to_expr(&self) -> Expr {
        use Expr::*;

        match (self.0.to_expr(), self.1.to_expr()) {
            (Zero, Zero) => Zero,
            (One, One) => Zero,

            (One, Zero) => One,
            (Zero, One) => Text(PreWrap {
                text: "-1".to_string(),
                needs_wrap: false,
            }),

            (Text(l), Zero) => Text(l),
            (Zero, Text(r)) => Text(PreWrap {
                text: format!("-{}", r.to_safe_string('(', ')')),
                needs_wrap: false,
            }),

            (Text(l), One) => Text(PreWrap {
                text: format!("{} - 1", l.to_safe_string('(', ')')),
                needs_wrap: true,
            }),
            (One, Text(r)) => Text(PreWrap {
                text: format!("1 - {}", r.to_safe_string('(', ')')),
                needs_wrap: true,
            }),
            (Text(l), Text(r)) => Text(PreWrap {
                text: format!("{} - {}", l.to_safe_string('(', ')'), r.to_safe_string('(', ')')),
                needs_wrap: true,
            })
        }
    }
}

impl<L: ToExpr, R: ToExpr> std::fmt::Display for Sub<L, R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.to_expr().fmt(f)
    }
}

/// Computes the (elementwise) product of two compatible [Buffer] types.
///
/// # Examples
/// ## x . y
/// ```
/// # #[macro_use] extern crate aegir;
/// # use aegir::{Identifier, Differentiable, Dual, buffers::Buffer, ops::Mul, ids::{X, Y}};
/// ctx!(Ctx { x: X, y: Y });
///
/// let f = Mul(X.into_var(), Y.into_var());
///
/// assert_eq!(f.evaluate_dual(X, &Ctx { x: 3.0, y: 2.0, }).unwrap(), dual!(6.0, 2.0));
/// assert_eq!(f.evaluate_dual(Y, &Ctx { x: 3.0, y: 2.0, }).unwrap(), dual!(6.0, 3.0));
/// ```
///
/// ## x . y^2
/// ```
/// # #[macro_use] extern crate aegir;
/// # use aegir::{Identifier, Node, Differentiable, Dual, buffers::Buffer, ops::Mul, ids::{X, Y}};
/// ctx!(Ctx { x: X, y: Y });
///
/// let f = Mul(X.into_var(), Y.into_var().pow(2.0f64.into_constant()));
///
/// assert_eq!(f.evaluate_dual(X, &Ctx { x: 3.0, y: 2.0, }).unwrap(), dual!(12.0, 4.0));
/// assert_eq!(f.evaluate_dual(Y, &Ctx { x: 3.0, y: 2.0, }).unwrap(), dual!(12.0, 12.0));
/// ```
#[derive(Copy, Clone, Debug, PartialEq, Contains)]
pub struct Mul<L, R>(#[op] pub L, #[op] pub R);

impl<L: Node, R: Node> Node for Mul<L, R> {}

impl<C, F, L, LV, R, RV, OS, OC, OV> Function<C> for Mul<L, R>
where
    C: Context,
    F: Scalar,

    L: Function<C, Value = LV>,
    LV: Buffer<Field = F> + ZipMap<RV, Output<F> = OV>,

    R: Function<C, Value = RV>,
    RV: Buffer<Field = F> + ZipMap<LV, Output<F> = OV>,

    OS: Shape,
    OC: Class<OS, Buffer<F> = OV>,
    OV: Buffer<Field = F, Class = OC, Shape = OS>,

    LV::Shape: shapes::Zip<RV::Shape, Shape = OS>,
{
    type Error = Error<C, L, R>;
    type Value = OV;

    fn evaluate<CR: AsRef<C>>(&self, ctx: CR) -> Result<Self::Value, Self::Error> {
        self.evaluate_spec(ctx).map(|state| state.unwrap())
    }

    fn evaluate_spec<CR: AsRef<C>>(&self, ctx: CR) -> Result<Spec<Self::Value>, Self::Error> {
        use Spec::*;

        let x = self.0.evaluate_spec(&ctx).map_err(BinaryError::Left)?;
        let y = self.1.evaluate_spec(&ctx).map_err(BinaryError::Right)?;

        match (x, y) {
            // If either value is zero, we can just short-circuit...
            (Full(sx, fx), Full(sy, fy)) if (fx * fy).is_zero() => {
                sx.zip(sy).map(Spec::zeroes).map_err(BinaryError::Output)
            },

            // If either value is one, we can just short-circuit...
            (Full(sx, fx), Full(sy, fy)) if fx.is_one() && fy.is_one() => {
                sx.zip(sy).map(Spec::ones).map_err(BinaryError::Output)
            },

            // If either value is unity, then we can also short-circuit...
            //  - In this case, the left-hand value is one, i.e. we have 1 * x = x.
            (Full(sx, fx), Raw(y)) if fx.is_one() => y
                .zip_map_dominate_id(sx)
                .map(Raw)
                .map_err(|err| err.reverse())
                .map_err(BinaryError::Output),
            //  - In this case, the right-hand value is one, i.e. we have x * 1 = x.
            (Raw(x), Full(sy, fy)) if fy.is_one() => x
                .zip_map_dominate_id(sy)
                .map(Raw)
                .map_err(BinaryError::Output),

            // Otherwise we just perform the full multiplication...
            (x, y) => x
                .unwrap()
                .zip_map(y.unwrap(), |xi, yi| xi * yi)
                .map(Raw)
                .map_err(BinaryError::Output),
        }
    }

    fn evaluate_shape<CR: AsRef<C>>(&self, ctx: CR) -> Result<ShapeOf<Self::Value>, Self::Error> {
        let l_shape = self.0.evaluate_shape(&ctx).map_err(BinaryError::Left)?;
        let r_shape = self.1.evaluate_shape(ctx).map_err(BinaryError::Right)?;

        l_shape.zip(r_shape).map_err(BinaryError::Output)
    }
}

impl<T, L, R> Differentiable<T> for Mul<L, R>
where
    T: Identifier,
    L: Differentiable<T> + Clone,
    R: Differentiable<T> + Clone,
{
    type Adjoint = Add<Mul<L::Adjoint, R>, Mul<R::Adjoint, L>>;

    fn adjoint(&self, target: T) -> Self::Adjoint {
        let gl = self.0.adjoint(target);
        let gr = self.1.adjoint(target);

        let ll = gl.mul(self.1.clone());
        let rr = gr.mul(self.0.clone());

        ll.add(rr)
    }
}

impl<L: ToExpr, R: ToExpr> ToExpr for Mul<L, R> {
    fn to_expr(&self) -> Expr {
        use Expr::*;

        match (self.0.to_expr(), self.1.to_expr()) {
            (_, Zero) | (Zero, _) => Zero,
            (One, One) => One,

            (l, One) => l,
            (One, r) => r,

            (Text(l), Text(r)) => Text(PreWrap {
                text: format!("{} \u{2218} {}", l.to_safe_string('(', ')'), r.to_safe_string('(', ')')),
                needs_wrap: true,
            })
        }
    }
}

impl<L: ToExpr, R: ToExpr> fmt::Display for Mul<L, R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.to_expr().fmt(f)
    }
}

/// Computes the (elementwise) quotient of two compatible [Buffer] types.
#[derive(Copy, Clone, Debug, PartialEq, Contains)]
pub struct Div<L, R>(#[op] pub L, #[op] pub R);

impl<L: Node, R: Node> Node for Div<L, R> {}

impl<C, F, L, LV, R, RV, OV> Function<C> for Div<L, R>
where
    C: Context,
    F: Scalar,

    L: Function<C, Value = LV>,
    LV: Buffer<Field = F> + ZipMap<RV, Output<F> = OV>,

    R: Function<C, Value = RV>,
    RV: Buffer<Field = F> + ZipMap<LV, Output<F> = OV>,

    OV: Buffer<Field = F>,

    LV::Shape: Zip<RV::Shape, Shape = OV::Shape>,
{
    type Error = Error<C, L, R>;
    type Value = OV;

    fn evaluate<CR: AsRef<C>>(&self, ctx: CR) -> Result<Self::Value, Self::Error> {
        let x = self.0.evaluate(ctx.as_ref()).map_err(BinaryError::Left)?;
        let y = self.1.evaluate(ctx).map_err(BinaryError::Right)?;

        x.zip_map(y, |xi, yi| xi / yi).map_err(BinaryError::Output)
    }

    fn evaluate_shape<CR: AsRef<C>>(&self, ctx: CR) -> Result<ShapeOf<Self::Value>, Self::Error> {
        let l_shape = self.0.evaluate_shape(&ctx).map_err(BinaryError::Left)?;
        let r_shape = self.1.evaluate_shape(ctx).map_err(BinaryError::Right)?;

        l_shape.zip(r_shape).map_err(BinaryError::Output)
    }
}

impl<T, L, R> Differentiable<T> for Div<L, R>
where
    T: Identifier,
    L: Differentiable<T> + Clone,
    R: Differentiable<T> + Clone,
{
    type Adjoint = Div<Sub<Mul<R, L::Adjoint>, Mul<L, R::Adjoint>>, crate::ops::Square<R>>;

    fn adjoint(&self, target: T) -> Self::Adjoint {
        let l = self.0.adjoint(target);
        let r = self.1.adjoint(target);

        let numerator = self.1.clone().mul(l).sub(self.0.clone().mul(r));

        numerator.div(self.1.clone().squared())
    }
}

impl<L: ToExpr, R: ToExpr> ToExpr for Div<L, R> {
    fn to_expr(&self) -> Expr {
        use Expr::*;

        match (self.0.to_expr(), self.1.to_expr()) {
            (Zero, _) => Zero,
            (l, One) => l,

            (_, Zero) => Text(PreWrap {
                text: "\u{221E}".to_string(),
                needs_wrap: false,
            }),

            (One, Text(r)) => Text(PreWrap {
                text: format!("1 / {}", r.to_safe_string('(', ')')),
                needs_wrap: true,
            }),

            (Text(l), Text(r)) => Text(PreWrap {
                text: format!("{} / {}", l.to_safe_string('(', ')'), r.to_safe_string('(', ')')),
                needs_wrap: true,
            })
        }
    }
}

impl<L: fmt::Display, R: Node + fmt::Display> fmt::Display for Div<L, R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}) / ({})", self.0, self.1)
    }
}
