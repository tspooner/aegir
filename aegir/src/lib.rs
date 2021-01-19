#[allow(unused_imports)]
#[macro_use]
extern crate aegir_derive;
#[doc(hidden)]
pub use self::aegir_derive::*;

#[allow(unused_imports)]
use paste::paste;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum NoError {}

impl std::fmt::Display for NoError {
    fn fmt(&self, _: &mut std::fmt::Formatter) -> std::fmt::Result { match *self {} }
}

impl std::error::Error for NoError {
    fn description(&self) -> &str { match *self {} }
}

/// Trait for type-level identifiers.
pub trait Identifier: Eq + Copy + std::fmt::Debug + std::fmt::Display {
    /// Convert the identifier into a [Variable](sources::Variable).
    fn to_var(self) -> sources::Variable<Self> { sources::Variable(self) }
}

// #[derive(Copy, Clone, Debug, PartialEq, Eq)]
// pub struct Indexed<T, Idx>(T, Idx);

// impl<T: Identifier, Idx> Indexed<T, Idx> {
// pub fn new(target: T, idx: Idx) -> Self { Indexed(target, idx) }
// }

// impl<T, Idx> std::fmt::Display for Indexed<T, Idx>
// where
// T: Identifier,
// Idx: std::fmt::Display,
// {
// fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
// write!(f, "{}[{}]", self.0, self.1)
// }
// }

// impl<T, Idx> Identifier for Indexed<T, Idx>
// where
// T: Identifier,
// Idx: Copy + Clone + Eq + std::fmt::Debug + std::fmt::Display,
// {}

/// Trait for types that store data.
pub trait Database {}

/// Trait for accessing entries in a [Database].
pub trait Get<T: Identifier>: Database {
    type Output;

    fn get(&self, target: T) -> Option<&Self::Output>;
}

#[derive(Copy, Clone)]
pub struct SimpleDatabase<T, B>(T, B);

impl<T: Identifier, B: buffer::Buffer> SimpleDatabase<T, B> {
    pub fn new(target: T, buffer: B) -> Self { SimpleDatabase(target, buffer) }
}

impl<T, B> Database for SimpleDatabase<T, B> {}

impl<T: Identifier, B: buffer::Buffer> Get<T> for SimpleDatabase<T, B> {
    type Output = B;

    fn get(&self, target: T) -> Option<&B> {
        if target == self.0 {
            Some(&self.1)
        } else {
            None
        }
    }
}

#[macro_export]
macro_rules! db {
    ($name:ident { $($entity_name:ident: $entity_type:ident),+ }) => {
        paste! {
            #[derive(Database)]
            pub struct $name<$([<__ $entity_type>]),+> {
                $(#[id($entity_type)] pub $entity_name: [<__ $entity_type>]),+
            }
        }
    }
}

#[macro_export]
macro_rules! simple_db {
    ($s:ident => $v:expr) => {
        SimpleDatabase::new($s, $v)
    };
}

/// Trait for computation nodes.
pub trait Node {
    fn named<I: Identifier>(self, id: I) -> NamedNode<Self, I>
    where
        Self: Sized,
    {
        NamedNode(self, id)
    }

    fn add<N: Node>(self, other: N) -> maths::Add<Self, N>
    where
        Self: Sized,
    {
        maths::Add(self, other)
    }

    fn sub<N: Node>(self, other: N) -> maths::Sub<Self, N>
    where
        Self: Sized,
    {
        maths::Sub(self, other)
    }

    fn mul<N: Node>(self, other: N) -> maths::Mul<Self, N>
    where
        Self: Sized,
    {
        maths::Mul(self, other)
    }

    fn dot<N: Node>(self, other: N) -> maths::InnerProduct<Self, N>
    where
        Self: Sized,
    {
        maths::InnerProduct::new(self, other)
    }

    fn abs(self) -> maths::Abs<Self>
    where
        Self: Sized,
    {
        maths::Abs(self)
    }

    fn neg(self) -> maths::Negate<Self>
    where
        Self: Sized,
    {
        maths::Negate(self)
    }

    fn pow<P>(self, power: P) -> maths::Power<Self, P>
    where
        Self: Sized,
    {
        maths::Power(self, power)
    }

    fn squared(self) -> maths::Square<Self>
    where
        Self: Sized,
    {
        maths::Square(self)
    }

    fn reduce(self) -> maths::Reduce<Self>
    where
        Self: Sized,
    {
        maths::Reduce(self)
    }
}

/// Trait for [Nodes](Node) that can assert their symbolic contents.
pub trait Contains<T: Identifier>: Node {
    fn contains(&self, target: T) -> bool;
}

/// Trait for [Nodes](Node) that can be evaluated against a database.
pub trait Function<D: Database>: Node {
    type Codomain: buffer::Buffer;
    type Error: std::error::Error;

    /// Evaluate the function and return the corresponding value in the
    /// [Codomain](Function::Codomain).
    ///
    /// # Examples
    /// ```
    /// # #[macro_use] extern crate aegir;
    /// # use aegir::{Identifier, SimpleDatabase, Function};
    /// # ids!(X);
    /// let db = SimpleDatabase::new(X, 1.0);
    ///
    /// assert_eq!(X.to_var().evaluate(&db).unwrap(), 1.0);
    /// ```
    fn evaluate(&self, db: &D) -> Result<Self::Codomain, Self::Error>;
}

/// Trait for [`Identifier`]-differentiable [Functions](Function).
pub trait Differentiable<D: Database, T: Identifier>: Function<D> + Contains<T> {
    type Jacobian: buffer::Buffer<Field = buffer::FieldOf<<Self as Function<D>>::Codomain>>;

    /// Compute the [Jacobian](Differentiable::Jacobian) associated with
    /// `target`.
    ///
    /// # Examples
    /// ```
    /// # #[macro_use] extern crate aegir;
    /// # use aegir::{Node, Identifier, SimpleDatabase, Function, Differentiable, buffer::Buffer};
    /// # ids!(X);
    /// let c = 2.0.into_constant();
    /// let db = SimpleDatabase::new(X, 10.0);
    ///
    /// assert_eq!(X.to_var().mul(c).grad(&db, X).unwrap(), 2.0);
    /// ```
    fn grad(&self, db: &D, target: T) -> Result<Self::Jacobian, Self::Error> {
        self.dual(db, target).map(|dual| dual.adjoint)
    }

    /// Evaluate the function and compute the
    /// [Jacobian](Differentiable::Jacobian) associated with `target` in a
    /// single pass.
    fn dual(
        &self,
        db: &D,
        target: T,
    ) -> Result<dual::Dual<Self::Codomain, Self::Jacobian>, Self::Error> {
        self.evaluate(db).and_then(|value| {
            self.grad(db, target)
                .map(|adjoint| dual::Dual { value, adjoint })
        })
    }
}

/// Trait for [Differentiables](Differentiable) that can be expressed statically
/// within `aegir`.
pub trait Compile<T: Identifier>: Node {
    type CompiledJacobian: Node;
    type Error: std::error::Error;

    /// Compile the [Jacobian](Differentiable::Jacobian) associated with
    /// `target`.
    ///
    /// # Examples
    /// ```
    /// # #[macro_use] extern crate aegir;
    /// # use aegir::{Node, Identifier, SimpleDatabase, Compile, Function, buffer::Buffer};
    /// # ids!(X);
    /// let c = 2.0f64.into_constant();
    /// let db = SimpleDatabase::new(X, 10.0);
    /// let grad = X.to_var().mul(c).compile_grad(X).unwrap();
    ///
    /// assert_eq!(grad.evaluate(&db).unwrap(), 2.0);
    /// ```
    fn compile_grad(&self, target: T) -> Result<Self::CompiledJacobian, Self::Error>;
}

pub type ErrorOf<F, D> = <F as Function<D>>::Error;
pub type CodomainOf<F, D> = <F as Function<D>>::Codomain;
pub type JacobianOf<F, D, T> = <F as Differentiable<D, T>>::Jacobian;
pub type DualOf<F, D, T> = Dual<
    <F as Function<D>>::Codomain,
    <F as Differentiable<D, T>>::Jacobian
>;

extern crate self as aegir;

#[macro_use]
mod macros;

mod dual;
pub use self::dual::Dual;

pub mod buffer;
pub mod maths;
pub mod sources;

mod named;
pub use self::named::NamedNode;

/// Evaluate a [Function] node against a given [Database].
pub fn evaluate<D, F>(f: F, db: &D) -> Result<F::Codomain, F::Error>
where
    F: Function<D>,
    D: Database,
{
    f.evaluate(db)
}

/// Compute the derivative of a [Differentiable] node wrt `target` against a given [Database].
pub fn differentiate<D, T, F>(f: F, db: &D, target: T) -> Result<F::Jacobian, F::Error>
where
    D: Database,
    T: Identifier,
    F: Differentiable<D, T>,
{
    f.grad(db, target)
}
