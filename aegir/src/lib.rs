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

pub trait Database {}

pub trait Get<T: Identifier>: Database {
    type Output;

    fn get(&self, target: T) -> Option<&Self::Output>;
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

pub trait Node {
    fn named<I: Identifier>(self, id: I) -> NamedNode<Self, I> where Self: Sized {
        NamedNode(self, id)
    }

    fn add<N: Node>(self, other: N) -> maths::arithmetic::Add<Self, N> where Self: Sized {
        maths::arithmetic::Add(self, other)
    }

    fn sub<N: Node>(self, other: N) -> maths::arithmetic::Sub<Self, N> where Self: Sized {
        maths::arithmetic::Sub(self, other)
    }

    fn mul<N: Node>(self, other: N) -> maths::arithmetic::Mul<Self, N> where Self: Sized {
        maths::arithmetic::Mul(self, other)
    }

    fn dot<N: Node>(self, other: N) -> maths::linalg::InnerProduct<Self, N> where Self: Sized {
        maths::linalg::InnerProduct::new(self, other)
    }

    fn abs(self) -> maths::arithmetic::Abs<Self> where Self: Sized {
        maths::arithmetic::Abs(self)
    }

    fn neg(self) -> maths::arithmetic::Neg<Self> where Self: Sized {
        maths::arithmetic::Neg(self)
    }

    fn pow<P>(self, power: P) -> maths::arithmetic::Power<Self, P> where Self: Sized {
        maths::arithmetic::Power(self, power)
    }

    fn squared(self) -> maths::arithmetic::Squared<Self> where Self: Sized {
        maths::arithmetic::Squared(self)
    }

    fn reduce(self) -> maths::reduce::Reduce<Self> where Self: Sized {
        maths::reduce::Reduce(self)
    }
}

pub trait Contains<T: Identifier>: Node {
    fn contains(&self, target: T) -> bool;
}

/// Trait for types that can be evaluated against a database.
pub trait Function<D: Database>: Node {
    type Codomain: buffer::Buffer;
    type Error: std::error::Error;

    fn evaluate(&self, db: &D) -> Result<Self::Codomain, Self::Error>;
}

pub trait Differentiable<D: Database, T: Identifier>: Function<D> + Contains<T> {
    type Jacobian: buffer::Buffer<
        Field = buffer::FieldOf<<Self as Function<D>>::Codomain>
    >;

    fn grad(&self, db: &D, target: T) -> Result<Self::Jacobian, Self::Error>;

    fn dual(&self, db: &D, target: T) -> Result<
        dual::Dual<Self::Codomain, Self::Jacobian>, Self::Error
    > {
        self.evaluate(db).and_then(|value| {
            self.grad(db, target).map(|adjoint| {
                dual::Dual {
                    value,
                    adjoint,
                }
            })
        })
    }
}

pub trait Compile<T: Identifier>: Node {
    type CompiledJacobian: Node;
    type Error: std::error::Error;

    fn compile_grad(&self, target: T) -> Result<Self::CompiledJacobian, Self::Error>;
}

pub trait Prune: Node where Self: Sized {
    fn prune(self) -> Option<Self>;
}

pub type ErrorOf<F, D> = <F as Function<D>>::Error;
pub type CodomainOf<F, D> = <F as Function<D>>::Codomain;
pub type JacobianOf<F, D, T> = <F as Differentiable<D, T>>::Jacobian;

extern crate self as aegir;

#[macro_use]
mod macros;

pub mod buffer;
pub mod dual;

pub mod sources;
pub mod maths;

mod named;
pub use self::named::NamedNode;

pub fn evaluate<D, F>(
    f: F,
    db: &D,
) -> Result<F::Codomain, F::Error>
where
    F: Function<D>,
    D: Database,
{
    f.evaluate(db)
}

pub fn differentiate<D, T, F>(
    f: F,
    db: &D,
    target: T,
) -> Result<F::Jacobian, F::Error>
where
    D: Database,
    T: Identifier,
    F: Differentiable<D, T>,
{
    f.grad(db, target)
}
