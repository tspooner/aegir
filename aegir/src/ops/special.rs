use crate::{
    buffers::{Buffer, Class, ClassOf, FieldOf, OwnedOf, ShapeOf},
    logic::TFU,
    Contains,
    Database,
    Function,
    Node,
    Stage,
};
use special_fun::FloatSpecial;
use std::fmt;

// Derive = x.gamma() * x.digamma()
#[derive(Clone, Copy, Debug, PartialEq, Contains)]
pub struct Gamma<N>(#[op] pub N);

impl<N: Node> Node for Gamma<N> {
    fn is_zero(_: Stage<&'_ Self>) -> TFU { TFU::False }
}

impl<D, N> Function<D> for Gamma<N>
where
    D: Database,
    N: Function<D>,

    FieldOf<N::Value>: special_fun::FloatSpecial,
{
    type Error = N::Error;
    type Value = OwnedOf<N::Value>;

    fn evaluate<DR: AsRef<D>>(&self, db: DR) -> Result<Self::Value, Self::Error> {
        self.0.evaluate(db).map(|buffer| buffer.map(|x| x.gamma()))
    }
}

impl<N: fmt::Display> fmt::Display for Gamma<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "\u{0393}({})", self.0) }
}

// Deriv = x.digamma()
#[derive(Clone, Copy, Debug, PartialEq, Contains)]
pub struct LogGamma<N>(#[op] pub N);

impl<N: Node> Node for LogGamma<N> {
    fn is_zero(_: Stage<&'_ Self>) -> TFU { TFU::False }
}

impl<D, N> Function<D> for LogGamma<N>
where
    D: Database,
    N: Function<D>,

    FieldOf<N::Value>: special_fun::FloatSpecial,
{
    type Error = N::Error;
    type Value = OwnedOf<N::Value>;

    fn evaluate<DR: AsRef<D>>(&self, db: DR) -> Result<Self::Value, Self::Error> {
        self.0
            .evaluate(db)
            .map(|buffer| buffer.map(|x| x.loggamma()))
    }
}

impl<N: fmt::Display> fmt::Display for LogGamma<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ln \u{0393}({})", self.0)
    }
}

// Deriv = x.loggamma()
#[derive(Clone, Copy, Debug, PartialEq, Contains)]
pub struct Factorial<N>(#[op] pub N);

impl<N: Node> Node for Factorial<N> {
    fn is_zero(_: Stage<&'_ Self>) -> TFU { TFU::False }
}

impl<D, N> Function<D> for Factorial<N>
where
    D: Database,
    N: Function<D>,

    FieldOf<N::Value>: special_fun::FloatSpecial,
{
    type Error = N::Error;
    type Value = OwnedOf<N::Value>;

    fn evaluate<DR: AsRef<D>>(&self, db: DR) -> Result<Self::Value, Self::Error> {
        let stage = crate::Stage::Evaluation(&self.0);

        if N::is_zero(stage) == crate::logic::TFU::True {
            self.0.evaluate_shape(db).map(|shape| {
                <ClassOf<Self::Value> as Class<ShapeOf<N::Value>, FieldOf<N::Value>>>::full(
                    shape,
                    num_traits::one(),
                )
            })
        } else {
            self.0
                .evaluate(db)
                .map(|buffer| buffer.map(|x| x.factorial()))
        }
    }
}

impl<N: fmt::Display> fmt::Display for Factorial<N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "{}!", self.0) }
}

#[derive(Clone, Copy, Debug, PartialEq, Contains)]
pub struct Erf<N>(#[op] pub N);

impl<N> Erf<N> {
    pub fn complementary(self) -> crate::ops::Negate<Self> { crate::ops::Negate(self) }
}

impl<N: Node> Node for Erf<N> {
    fn is_zero(stage: Stage<&'_ Self>) -> TFU { stage.map(|node| &node.0).is_zero() }
}

impl<D, N> Function<D> for Erf<N>
where
    D: Database,
    N: Function<D>,

    crate::buffers::FieldOf<N::Value>: special_fun::FloatSpecial,
{
    type Error = N::Error;
    type Value = crate::buffers::OwnedOf<N::Value>;

    fn evaluate<DR: AsRef<D>>(&self, db: DR) -> Result<Self::Value, Self::Error> {
        let stage = crate::Stage::Evaluation(&self.0);

        if N::is_zero(stage) == crate::logic::TFU::True {
            self.0.evaluate_shape(db).map(|shape| {
                <ClassOf<Self::Value> as Class<ShapeOf<N::Value>, FieldOf<N::Value>>>::full(
                    shape,
                    num_traits::zero(),
                )
            })
        } else {
            self.0.evaluate(db).map(|buffer| buffer.map(|x| x.erf()))
        }
    }
}

impl<X: fmt::Display + PartialEq> fmt::Display for Erf<X> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "erf({})", self.0) }
}
