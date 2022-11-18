use crate::buffers::{Class, BufferOf, shapes::Shape, Scalars, Vecs, Tuples, Arrays};

pub struct Prec;

pub trait Precedence<C1, C2> {
    type Class;
}

macro_rules! impl_class_precedence {
    (($cl1:ty, $cl2:ty) => $cl3:ty) => {
        impl Precedence<$cl1, $cl2> for Prec {
            type Class = $cl3;
        }
    };
}

impl_class_precedence!((Scalars, Scalars) => Scalars);
impl_class_precedence!((Scalars, Arrays) => Arrays);
impl_class_precedence!((Scalars, Tuples) => Tuples);
impl_class_precedence!((Scalars, Vecs) => Vecs);

impl_class_precedence!((Arrays, Scalars) => Arrays);
impl_class_precedence!((Arrays, Arrays) => Arrays);
impl_class_precedence!((Arrays, Tuples) => Arrays);
impl_class_precedence!((Arrays, Vecs) => Arrays);

impl_class_precedence!((Tuples, Scalars) => Tuples);
impl_class_precedence!((Tuples, Arrays) => Arrays);
impl_class_precedence!((Tuples, Tuples) => Tuples);
impl_class_precedence!((Tuples, Vecs) => Vecs);

impl_class_precedence!((Vecs, Scalars) => Vecs);
impl_class_precedence!((Vecs, Arrays) => Vecs);
impl_class_precedence!((Vecs, Tuples) => Vecs);
impl_class_precedence!((Vecs, Vecs) => Vecs);

/// Error type for variable/source nodes.
#[derive(Copy, Clone, Debug)]
pub enum SourceError<ID> {
    /// Error case when the `ID` variable is undefined.
    Undefined(ID),
}

impl<ID: std::fmt::Debug> std::fmt::Display for SourceError<ID> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SourceError::Undefined(id) => {
                write!(f, "No variable found with identifier {:?}.", id)
            },
        }
    }
}

impl<ID: std::fmt::Debug> std::error::Error for SourceError<ID> {}

mod constant;
pub use self::constant::{Constant, ConstantAdjoint};

mod variable;
pub use self::variable::{Variable, VariableAdjoint};
