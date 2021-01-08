#[allow(unused_imports)]
#[macro_use]
extern crate aegir_derive;
#[doc(hidden)]
pub use self::aegir_derive::*;

#[allow(unused_imports)]
use paste::paste;

pub trait Identifier: Eq + Copy + std::fmt::Debug + std::fmt::Display {
    fn to_var(self) -> sources::Variable<Self> { sources::Variable(self) }
}

pub trait State {}

pub trait Get<T: Identifier>: State {
    type Output;

    fn get(&self, target: T) -> Option<&Self::Output>;
}

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

pub trait Node {
    fn named<I: Identifier>(self, id: I) -> NamedNode<Self, I> where Self: Sized {
        NamedNode(self, id)
    }

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

pub trait Contains<T: Identifier>: Node {
    fn contains(&self, target: T) -> bool;
}

pub trait Function<S: State>: Node {
    type Codomain: buffer::Buffer;
    type Error: std::error::Error;

    fn evaluate(&self, state: &S) -> Result<Self::Codomain, Self::Error>;
}

pub trait Differentiable<S: State, T: Identifier>: Function<S> + Contains<T> {
    type Jacobian: buffer::Buffer<
        Field = buffer::FieldOf<<Self as Function<S>>::Codomain>
    >;

    fn grad(&self, state: &S, target: T) -> Result<Self::Jacobian, Self::Error>;

    fn dual(&self, state: &S, target: T) -> Result<
        dual::Dual<Self::Codomain, Self::Jacobian>, Self::Error
    > {
        self.evaluate(state).and_then(|value| {
            self.grad(state, target).map(|adjoint| {
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
pub type JacobianOf<F, S, T> = <F as Differentiable<S, T>>::Jacobian;

pub mod buffer;
pub mod dual;

pub mod sources;
pub mod ops;

mod named;
pub use self::named::NamedNode;

// pub mod module;

pub fn evaluate<S, F>(
    f: F,
    state: &S,
) -> Result<F::Codomain, F::Error>
where
    S: State,
    F: Function<S>,
{
    f.evaluate(state)
}

pub fn derivative<S, T, F>(
    f: F,
    target: T,
    state: &S,
) -> Result<F::Jacobian, F::Error>
where
    S: State,
    T: Identifier,
    F: Differentiable<S, T>,
{
    f.grad(state, target)
}
