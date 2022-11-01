use super::*;

/// Trait used for defining precedence relations between [class](Class) pairs.
///
/// A precedence mapping is defined as a function from the set of
/// class-class-shape triples to the set of types implementing `Class<S, F>`. In
/// other words, take two class types, `C1` and `C2`, with the same field `F`
/// and shapes `S1` and `S2`, respectively. Then, given a desired shape `S`, the
/// associated type [Precedence::Class] is given by `C1` or `C2` iff it
/// implements [Class] for `F` and `S`.
pub trait Precedence<C: Class<S>, S: Shape>: Class<S> {
    /// Class with precedence over `Self::Class` and `B::Class` and shape `S`.
    type Class: Class<S>;
}

pub type PClassOf<C1, C2, S> = <C1 as Precedence<C2, S>>::Class;

pub type PBufferOf<C1, C2, S, F> = BufferOf<PClassOf<C1, C2, S>, S, F>;

pub fn build<C1, C2, S, F>(shape: S, f: impl Fn(S::Index) -> F) -> BufferOf<C1::Class, S, F>
where
    C1: Precedence<C2, S>,
    C2: Class<S>,

    S: Shape,
    F: Scalar,
{
    <PClassOf<C1, C2, S> as Class<S>>::build(shape, f)
}

pub fn build_subset<C1, C2, S, F, Iter, Func>(
    shape: S,
    base: F,
    subset: Iter,
    active: Func,
) -> BufferOf<C1::Class, S, F>
where
    C1: Precedence<C2, S>,
    C2: Class<S>,

    S: Shape,
    F: Scalar,

    Iter: Iterator<Item = shapes::IndexOf<S>>,
    Func: Fn(S::Index) -> F,
{
    <PClassOf<C1, C2, S> as Class<S>>::build_subset(shape, base, subset, active)
}

pub fn full<C1, C2, S, F>(shape: S, value: F) -> BufferOf<C1::Class, S, F>
where
    C1: Precedence<C2, S>,
    C2: Class<S>,

    S: Shape,
    F: Scalar,
{
    <PClassOf<C1, C2, S> as Class<S>>::full(shape, value)
}

pub fn diagonal<C1, C2, S, F>(shape: S, value: F) -> BufferOf<C1::Class, S, F>
where
    C1: Precedence<C2, S>,
    C2: Class<S>,

    S: Shape,
    F: Scalar,
{
    <PClassOf<C1, C2, S> as Class<S>>::diagonal(shape, value)
}

pub fn identity<C1, C2, S, F>(shape: S) -> BufferOf<C1::Class, S, F>
where
    C1: Precedence<C2, S>,
    C2: Class<S>,

    S: Shape,
    F: Scalar,
{
    <PClassOf<C1, C2, S> as Class<S>>::identity(shape)
}

macro_rules! impl_class_precedence {
    (($cl1:ty, $cl2:ty) => $cl3:ty) => {
        impl<S: Shape> Precedence<$cl2, S> for $cl1
        where
            $cl1: Class<S>,
            $cl2: Class<S>,
        {
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
