use crate::{
    buffers::{
        shapes::{Concat, Indices, Shape},
        Buffer,
        Class,
        FieldOf,
        OwnedOf,
        Precedence,
        PrecedenceMapping,
        PrecedenceOf,
        Scalar,
    },
    Contains,
    Database,
    Differentiable,
    Function,
    Identifier,
    Node,
    Read,
};

#[derive(Copy, Clone, Debug)]
pub enum VariableError<ID> {
    Undefined(ID),
}

impl<ID: std::fmt::Debug> std::fmt::Display for VariableError<ID> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VariableError::Undefined(id) => {
                write!(f, "No variable found with identifier {:?}.", id)
            },
        }
    }
}

impl<ID: std::fmt::Debug> std::error::Error for VariableError<ID> {}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Constant<B>(pub B);

impl<B> Node for Constant<B> {}

impl<T, B> Contains<T> for Constant<B>
where
    T: Identifier,
{
    fn contains(&self, _: T) -> bool { false }
}

impl<D, B> Function<D> for Constant<B>
where
    D: Database,
    B: Buffer,
{
    type Error = VariableError<()>;
    type Value = OwnedOf<B>;

    fn evaluate<DR: AsRef<D>>(&self, _: DR) -> Result<Self::Value, Self::Error> {
        Ok(self.0.to_owned())
    }
}

impl<T, B> Differentiable<T> for Constant<B>
where
    T: Identifier,
    B: Buffer,

    FieldOf<B>: num_traits::Zero,
{
    type Adjoint = Constant<OwnedOf<B>>;

    fn adjoint(&self, _: T) -> Self::Adjoint { Constant(self.0.to_zeroes()) }
}

impl<B: std::fmt::Display> std::fmt::Display for Constant<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { self.0.fmt(f) }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Variable<I>(pub I);

impl<I> Node for Variable<I> {}

impl<T, I> Contains<T> for Variable<I>
where
    T: Identifier,
    I: Identifier + std::cmp::PartialEq<T>,
{
    fn contains(&self, ident: T) -> bool { self.0 == ident }
}

impl<D, I> Function<D> for Variable<I>
where
    D: Read<I>,
    I: Identifier,
{
    type Error = VariableError<I>;
    type Value = OwnedOf<D::Buffer>;

    fn evaluate<DR: AsRef<D>>(&self, db: DR) -> Result<Self::Value, Self::Error> {
        db.as_ref()
            .read(self.0)
            .map(|v| v.to_owned())
            .ok_or_else(|| VariableError::Undefined(self.0))
    }
}

// impl<S1, S2, S3, F, C, I, T, BI, BT, D> Differentiable<D, T> for Variable<I>
// where
// S1: shapes::Concat<S2, Shape = S3>,
// S2: shapes::Shape,
// S3: shapes::Split<Right = <S3 as shapes::Split>::Left>,

// S3::Left: shapes::Indices + shapes::Concat<S3::Left, Shape = S3>,

// F: Scalar,
// C: Class<S1, F> + Class<S2, F> + Class<S3, F>,
// T: Identifier,
// I: Identifier + std::cmp::PartialEq<T>,

// BI: Buffer<Class = C, Shape = S1, Field = F>,
// BT: Buffer<Class = C, Shape = S2, Field = F>,

// D: Read<T, Buffer = BT> + Read<I, Buffer = BI>,
// {
// type Adjoint = <C as Class<S3, F>>::Buffer;

// fn grad(&self, db: &D, ident: T) -> Result<Self::Adjoint, Self::Error> {
// let shape_inner = db.get_shape(self.0).ok_or(VariableError(self.0))?;
// let shape_ident = db.get_shape(ident).unwrap();
// let shape_output = shape_inner.concat(shape_target);

// if self.contains(target) {
// db.get(self.0)
// .map(|_| {
// let ixs = shape_output.split().0.indices().map(|ix|
// <S3::Left as shapes::Concat>::concat_indices(ix.clone(), ix)
// );

// <C as Class<S3, F>>::full_indices(
// shape_output, num_traits::zero(), ixs, num_traits::one()
// )
// })
// .ok_or_else(|| VariableError(self.0))
// } else {
// db.get(self.0)
// .map(|_| {
// <C as Class<S3, F>>::full(shape_output, num_traits::zero())
// })
// .ok_or_else(|| VariableError(self.0))
// }
// }
// }

impl<I, T> Differentiable<T> for Variable<I>
where
    I: Identifier,
    T: Identifier,
{
    type Adjoint = VariableAdjoint<I, T>;

    fn adjoint(&self, target: T) -> Self::Adjoint {
        VariableAdjoint {
            value: self.0,
            target: target,
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

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct VariableAdjoint<I, T> {
    pub value: I,
    pub target: T,
}

impl<I, T> Node for VariableAdjoint<I, T> {}

impl<I, T, A> Contains<A> for VariableAdjoint<I, T>
where
    I: Identifier + PartialEq<A>,
    T: Identifier + PartialEq<A>,
    A: Identifier,
{
    fn contains(&self, ident: A) -> bool { self.value == ident || self.target == ident }
}

impl<I, T, D, SI, CI, ST, CT, SA, F> Function<D> for VariableAdjoint<I, T>
where
    I: Identifier + PartialEq<T>,
    T: Identifier,

    D: Read<I> + Read<T>,

    SI: Concat<SI> + Concat<ST, Shape = SA> + Indices,
    ST: Shape + Indices,
    SA: Shape,

    F: Scalar,

    CI: Class<SI, F>,
    CT: Class<ST, F>,
    Precedence: PrecedenceMapping<F, SI, CI, ST, CT>,

    <D as Read<I>>::Buffer: Buffer<Class = CI, Shape = SI, Field = F>,
    <D as Read<T>>::Buffer: Buffer<Class = CT, Shape = ST, Field = F>,
{
    type Error = crate::BinaryError<VariableError<I>, VariableError<T>, crate::NoError>;
    type Value = <PrecedenceOf<F, SI, CI, ST, CT> as Class<SA, F>>::Buffer;

    fn evaluate<DR: AsRef<D>>(&self, db: DR) -> Result<Self::Value, Self::Error> {
        let shape_value = db.as_ref().read(self.value).map(|buf| buf.shape()).ok_or(
            crate::BinaryError::Left(VariableError::Undefined(self.value)),
        )?;
        let shape_target = db.as_ref().read(self.target).map(|buf| buf.shape()).ok_or(
            crate::BinaryError::Right(VariableError::Undefined(self.target)),
        )?;
        let shape_adjoint = shape_value.concat(shape_target);

        Ok(if self.value == self.target {
            // In this case, we also know that shape_value == shape_target.
            // This further implies that shape_adjoint.split() is exactly equal
            // to (shape_value, shape_adjoint). We exploit this below:
            let ixs = shape_value
                .indices()
                .zip(shape_target.indices())
                .map(|ixs| <SI as Concat<ST>>::concat_indices(ixs.0, ixs.1));

            let one: F = num_traits::one();

            <PrecedenceOf<F, SI, CI, ST, CT> as Class<SA, F>>::build_subset(
                shape_adjoint,
                num_traits::zero(),
                ixs,
                |_| one,
            )
        } else {
            <Self::Value as Buffer>::Class::full(shape_adjoint, num_traits::zero())
        })
    }
}

impl<I, T, A> Differentiable<A> for VariableAdjoint<I, T>
where
    A: Identifier,

    Self: Clone,
{
    type Adjoint = ZeroedAdjoint<Self, A>;

    fn adjoint(&self, ident: A) -> Self::Adjoint {
        ZeroedAdjoint {
            node: self.clone(),
            target: ident,
        }
    }
}

impl<I, T> std::fmt::Display for VariableAdjoint<I, T>
where
    I: std::fmt::Display + PartialEq<T>,
    T: std::fmt::Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.value == self.target {
            write!(f, "\u{2202}{}", self.value)
        } else {
            write!(f, "0")
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct ZeroedAdjoint<N, T> {
    pub node: N,
    pub target: T,
}

impl<N: Node, T> Node for ZeroedAdjoint<N, T> {}

impl<N, T, A> Contains<A> for ZeroedAdjoint<N, T>
where
    N: Contains<A>,
    T: Identifier + PartialEq<A>,
    A: Identifier,
{
    fn contains(&self, ident: A) -> bool { self.node.contains(ident) || self.target == ident }
}

impl<N, T, D, F, SN, CN, ST, CT, SA> Function<D> for ZeroedAdjoint<N, T>
where
    F: Scalar,

    SN: Concat<ST, Shape = SA>,
    ST: Shape,
    SA: Shape,

    CN: Class<SN, F>,
    CT: Class<ST, F>,
    Precedence: PrecedenceMapping<F, SN, CN, ST, CT>,

    N: Function<D>,
    N::Value: Buffer<Field = F, Shape = SN, Class = CN>,

    T: Identifier,

    D: Read<T>,
    D::Buffer: Buffer<Field = F, Shape = ST, Class = CT>,
{
    type Error = crate::BinaryError<N::Error, VariableError<T>, crate::NoError>;
    type Value = <PrecedenceOf<F, SN, CN, ST, CT> as Class<SA, F>>::Buffer;

    fn evaluate<DR: AsRef<D>>(&self, db: DR) -> Result<Self::Value, Self::Error> {
        let shape_value = self
            .node
            .evaluate_shape(db.as_ref())
            .map_err(crate::BinaryError::Left)?;
        let shape_target = db.as_ref().read(self.target).map(|buf| buf.shape()).ok_or(
            crate::BinaryError::Right(VariableError::Undefined(self.target)),
        )?;
        let shape_adjoint = shape_value.concat(shape_target);

        Ok(<PrecedenceOf<F, SN, CN, ST, CT> as Class<SA, F>>::full(
            shape_adjoint,
            num_traits::zero(),
        ))
    }
}

impl<N, T> std::fmt::Display for ZeroedAdjoint<N, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { write!(f, "0") }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aegir::{
        buffers::Buffer,
        ids::{X, Y},
        Function,
        Identifier,
    };

    #[derive(Database)]
    struct DB<A, B> {
        #[id(X)]
        pub x: A,

        #[id(Y)]
        pub y: B,
    }

    #[test]
    fn test_variable() {
        let var = X.into_var();

        assert_eq!(var.evaluate(&DB { x: 1.0, y: 0.0 }).unwrap(), 1.0);
        assert_eq!(
            var.evaluate(&DB {
                x: [-10.0, 5.0],
                y: 0.0
            })
            .unwrap(),
            [-10.0, 5.0]
        );
        assert_eq!(
            var.evaluate(&DB {
                x: (-1.0, 50.0),
                y: 0.0
            })
            .unwrap(),
            (-1.0, 50.0)
        );
        assert_eq!(
            var.evaluate(&DB {
                x: vec![1.0, 2.0],
                y: 0.0
            })
            .unwrap(),
            vec![1.0, 2.0]
        );
    }

    #[test]
    fn test_constant() {
        let c = 2.0f64.into_constant();

        assert_eq!(c.evaluate(&DB { x: 1.0, y: 0.0 }).unwrap(), 2.0);
        assert_eq!(
            c.evaluate(&DB {
                x: [-10.0, 5.0],
                y: 0.0
            })
            .unwrap(),
            2.0
        );
        assert_eq!(
            c.evaluate(&DB {
                x: (-1.0, 50.0),
                y: 0.0
            })
            .unwrap(),
            2.0
        );
        assert_eq!(
            c.evaluate(&DB {
                x: vec![1.0, 2.0],
                y: 0.0
            })
            .unwrap(),
            2.0
        );
    }

    #[test]
    fn test_adjoint_zero() {
        let g = X.into_var().adjoint(Y);

        assert_eq!(g.evaluate(&DB { x: 1.0, y: 0.0 }).unwrap(), 0.0);
        // assert_eq!(g.evaluate(&DB { x: [-10.0, 5.0], y: 0.0 }).unwrap(),
        // [0.0; 2]); assert_eq!(g.evaluate(&DB { x: (-1.0, 50.0), y:
        // 0.0 }).unwrap(), (0.0, 0.0)); assert_eq!(g.evaluate(&DB { x:
        // vec![1.0, 2.0], y: 0.0 }).unwrap(), vec![0.0; 2]);
    }

    #[test]
    fn test_adjoint_one() {
        let g = X.into_var().adjoint(X);

        assert_eq!(g.evaluate(&DB { x: 1.0, y: 0.0 }).unwrap(), 1.0);
        // assert_eq!(g.evaluate(&DB { x: [-10.0, 5.0] }).unwrap(), [1.0; 2]);
        // assert_eq!(g.evaluate(&DB { x: (-1.0, 50.0) }).unwrap(), (1.0, 1.0));
        // assert_eq!(g.evaluate(&DB { x: vec![1.0, 2.0] }).unwrap(), vec![1.0;
        // 2]);
    }
}
