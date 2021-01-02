#[allow(unused_imports)]
// #[macro_use]
// extern crate aegir_derive;
// #[doc(hidden)]
// pub use self::aegir_derive::*;

pub trait Identifier: Eq + Copy + std::fmt::Debug + std::fmt::Display {}

#[macro_export]
macro_rules! id {
    ($name:ident::$symbol:expr) => {
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        pub struct $name;

        impl $crate::Identifier for $name {}

        impl std::fmt::Display for $name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, stringify!($symbol))
            }
        }
    };
    ($name:ident) => { id!{$name::stringify!($name)} }
}

#[macro_export]
macro_rules! ids {
    ($($name:ident::$symbol:expr),*) => { $($crate::id!($name::$symbol);)* };
    ($name:ident) => { $($crate::id!($name);)* }
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

pub type CodomainOf<F, S> = <F as Function<S>>::Codomain;
pub type ErrorOf<F, S> = <F as Function<S>>::Error;
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
