#[allow(unused_imports)]
#[macro_use]
extern crate aegir_derive;
#[doc(hidden)]
pub use self::aegir_derive::*;

#[allow(unused_imports)]
use paste::paste;

#[macro_export]
macro_rules! state {
    ($name:ident { $($entity_name:ident: $entity_type:ident),+ }) => {
        paste! {
            #[derive(State)]
            pub struct $name<$([<__ $entity_type>]),+> {
                $(#[id($entity_type)] pub $entity_name: [<__ $entity_type>]),+
            }
        }
    }
}

pub trait Identifier: Eq + Copy + std::fmt::Debug + std::fmt::Display {
    fn to_var(self) -> sources::Variable<Self> { sources::Variable(self) }
}

#[derive(Copy, Clone, Debug)]
pub struct GetError<ID>(ID);

impl<ID: std::fmt::Display> std::fmt::Display for GetError<ID> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Failed to get {} from data source.", self.0)
    }
}

impl<ID: std::fmt::Debug + std::fmt::Display> std::error::Error for GetError<ID> {}

pub trait Get<ID: Identifier> {
    type Output;

    fn get(&self, id: ID) -> Result<&Self::Output, GetError<ID>>;
}

pub trait GetMut<ID: Identifier>: Get<ID> {
    fn get_mut(&mut self, id: ID) -> &mut Self::Output;
}

pub trait Map<ID: Identifier, B: buffer::Buffer>: Get<ID> {
    type Output: Get<ID, Output = B>;

    fn map(
        self,
        id: ID,
        func: impl Fn(<Self as Get<ID>>::Output) -> B
    ) -> <Self as Map<ID, B>>::Output;
}

pub trait Node {
    // fn contains<T: Identifier>(&self, target: T) -> bool;

    fn add<N: Node>(self, other: N) -> ops::scalar::Add<Self, N> where Self: Sized {
        ops::scalar::Add(self, other)
    }

    fn sub<N: Node>(self, other: N) -> ops::scalar::Sub<Self, N> where Self: Sized {
        ops::scalar::Sub(self, other)
    }

    fn mul<N: Node>(self, other: N) -> ops::scalar::Mul<Self, N> where Self: Sized {
        ops::scalar::Mul(self, other)
    }

    fn dot<N: Node>(self, other: N) -> ops::linalg::InnerProduct<Self, N> where Self: Sized {
        ops::linalg::InnerProduct::new(self, other)
    }

    fn abs(self) -> ops::scalar::Abs<Self> where Self: Sized { ops::scalar::Abs(self) }

    fn neg(self) -> ops::scalar::Neg<Self> where Self: Sized { ops::scalar::Neg(self) }

    fn pow<P>(self, power: P) -> ops::scalar::Power<Self, P> where Self: Sized {
        ops::scalar::Power(self, power)
    }

    fn reduce(self) -> ops::reduce::Reduce<Self> where Self: Sized { ops::reduce::Reduce(self) }
}

pub trait Function<S>: Node {
    type Codomain: buffer::Buffer;
    type Error: std::error::Error;

    fn evaluate(&self, state: &S) -> Result<Self::Codomain, Self::Error>;
}

pub trait Differentiable<T: Identifier, S>: Function<S> {
    type Jacobian: buffer::Buffer<
        Field = buffer::FieldOf<<Self as Function<S>>::Codomain>
    >;

    fn grad(&self, target: T, state: &S) -> Result<Self::Jacobian, Self::Error>;

    fn dual(&self, target: T, state: &S) -> Result<
        dual::Dual<Self::Codomain, Self::Jacobian>, Self::Error
    > {
        self.evaluate(state).and_then(|value| {
            self.grad(target, state).map(|adjoint| {
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

pub type ErrorOf<F, S> = <F as Function<S>>::Error;
pub type CodomainOf<F, S> = <F as Function<S>>::Codomain;
pub type JacobianOf<F, T, S> = <F as Differentiable<T, S>>::Jacobian;

pub mod buffer;
pub mod dual;

pub mod sources;
pub mod ops;

// pub mod module;

pub fn evaluate<S, F: Function<S>>(
    f: F,
    state: &S,
) -> Result<F::Codomain, F::Error>
{
    f.evaluate(state)
}

pub fn derivative<T: Identifier, S, F: Differentiable<T, S>>(
    f: F,
    target: T,
    state: &S,
) -> Result<F::Jacobian, F::Error>
{
    f.grad(target, state)
}
