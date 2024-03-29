use super::*;

/// Fixed 0-dimensional (scalar) shape.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct S0;

impl Shape for S0 {
    type Index = ();
    type IndexIter = std::iter::Once<()>;

    const DIM: usize = 0;

    fn contains(&self, _: ()) -> bool { true }

    fn cardinality(&self) -> usize { 1 }

    fn indices(&self) -> std::iter::Once<()> { std::iter::once(()) }
}

impl Broadcast for S0 {
    type Shape = S0;

    #[inline]
    fn broadcast(self, _: S0) -> Result<S0, IncompatibleShapes<S0>> { Ok(self) }
}

impl std::fmt::Display for S0 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { write!(f, "S0") }
}

/// Fixed 1-dimensional (vector) shape.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct S1<const A: usize>;

impl<const A: usize> Shape for S1<A> {
    type Index = usize;
    type IndexIter = std::ops::Range<usize>;

    const DIM: usize = 1;

    fn contains(&self, ix: usize) -> bool { ix < A }

    fn cardinality(&self) -> usize { A }

    fn indices(&self) -> std::ops::Range<usize> { 0..A }
}

impl<const A: usize> Broadcast for S1<A> {
    type Shape = S1<A>;

    #[inline]
    fn broadcast(self, _: S1<A>) -> Result<S1<A>, IncompatibleShapes<S1<A>>> { Ok(self) }
}

impl<const A: usize> Broadcast<S0> for S1<A> {
    type Shape = S1<A>;

    #[inline]
    fn broadcast(self, _: S0) -> Result<S1<A>, IncompatibleShapes<S1<A>, S0>> { Ok(self) }
}

impl<const A: usize> Broadcast<S1<A>> for S0 {
    type Shape = S1<A>;

    #[inline]
    fn broadcast(self, rhs: S1<A>) -> Result<S1<A>, IncompatibleShapes<S0, S1<A>>> { Ok(rhs) }
}

impl<const A: usize> std::fmt::Display for S1<A> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { write!(f, "{}", A) }
}

macro_rules! impl_fixed {
    (@dim_string; $fmt:ident :: $tp0:ident, $($tp:ident),*) => {{
        let mut res = write!($fmt, "{}", $tp0);

        $({
            res = res.and(write!($fmt, " \u{00D7} {}", $tp));
        })*

        res
    }};
    ($dim:literal; $name:ident<$($tp:ident),*>; $trans:expr) => {
        impl<$(const $tp: usize),+> Shape for $name<$($tp),+> {
            const DIM: usize = $dim;

            type Index = [usize; $dim];
            type IndexIter = Box<dyn Iterator<Item = Self::Index>>;

            fn contains(&self, ix: [usize; $dim]) -> bool {
                IntoIterator::into_iter(ix)
                    .zip(IntoIterator::into_iter([$($tp),+]))
                    .all(|(l, r)| l < r)
            }

            fn cardinality(&self) -> usize {
                IntoIterator::into_iter([$($tp),+]).product()
            }

            fn indices(&self) -> Self::IndexIter {
                Box::new(iproduct!($(0..$tp),+).map($trans))
            }
        }

        impl<$(const $tp: usize),+> Broadcast for $name<$($tp),+> {
            type Shape = Self;

            #[inline]
            fn broadcast(self, _: Self) -> Result<Self, IncompatibleShapes<Self>> { Ok(self) }
        }

        impl<$(const $tp: usize),+> Broadcast<S0> for $name<$($tp),+> {
            type Shape = Self;

            #[inline]
            fn broadcast(self, _: S0) -> Result<Self, IncompatibleShapes<Self, S0>> { Ok(self) }
        }

        impl<$(const $tp: usize),+> Broadcast<$name<$($tp),+>> for S0 {
            type Shape = $name<$($tp),+>;

            #[inline]
            fn broadcast(self, rhs: $name<$($tp),+>) -> Result<$name<$($tp),+>, IncompatibleShapes<S0, $name<$($tp),+>>> { Ok(rhs) }
        }

        impl<$(const $tp: usize),+> std::fmt::Display for $name<$($tp),+> {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                impl_fixed!(@dim_string; f :: $($tp),+)
            }
        }
    };
}

/// Fixed 2-dimensional (matrix) shape.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct S2<const A: usize, const B: usize>;

impl_fixed!(2; S2<A, B>; |ix| [ix.0, ix.1]);

/// Fixed 3-dimensional shape.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct S3<const A: usize, const B: usize, const C: usize>;

impl_fixed!(3; S3<A, B, C>; |ix| [ix.0, ix.1, ix.2]);

/// Fixed 4-dimensional shape.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct S4<const A: usize, const B: usize, const C: usize, const D: usize>;

impl_fixed!(4; S4<A, B, C, D>; |ix| [ix.0, ix.1, ix.2, ix.3]);

/// Fixed 5-dimensional shape.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct S5<const A: usize, const B: usize, const C: usize, const D: usize, const E: usize>;

impl_fixed!(5; S5<A, B, C, D, E>; |ix| [ix.0, ix.1, ix.2, ix.3, ix.4]);

/// Fixed 6-dimensional shape.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct S6<
    const A: usize,
    const B: usize,
    const C: usize,
    const D: usize,
    const E: usize,
    const F: usize,
>;

impl_fixed!(6; S6<A, B, C, D, E, F>; |ix| [ix.0, ix.1, ix.2, ix.3, ix.4, ix.5]);

impl<const A: usize, const B: usize> Split for S2<A, B> {
    type Left = S1<A>;
    type Right = S1<B>;

    fn split(self) -> (S1<A>, S1<B>) { (S1, S1) }

    fn split_index(index: [usize; 2]) -> (usize, usize) { (index[0], index[1]) }
}

impl<const A: usize, const B: usize, const C: usize, const D: usize> Split for S4<A, B, C, D> {
    type Left = S2<A, B>;
    type Right = S2<C, D>;

    fn split(self) -> (S2<A, B>, S2<C, D>) { (S2, S2) }

    fn split_index(index: [usize; 4]) -> ([usize; 2], [usize; 2]) {
        ([index[0], index[1]], [index[2], index[3]])
    }
}

macro_rules! impl_concat {
    (S0 + $right:ident<$($rp:ident::$ri:literal),+>) => {
        impl<$(const $rp: usize),+> Concat<$right<$($rp),+>> for S0 {
            type Shape = $right<$($rp),+>;

            fn concat(self, _: $right<$($rp),+>) -> Self::Shape { $right }

            fn concat_indices(
                _: Self::Index,
                r: IndexOf<$right<$($rp),+>>
            ) -> IndexOf<Self::Shape>
            {
                [$(r[$ri]),+]
            }
        }
    };
    ($left:ident<$($lp:ident::$li:literal),+> + S0) => {
        impl<$(const $lp: usize),+> Concat<S0> for $left<$($lp),+> {
            type Shape = $left<$($lp),+>;

            fn concat(self, _: S0) -> Self::Shape { $left }

            fn concat_indices(l: IndexOf<$left<$($lp),+>>, _: ()) -> IndexOf<Self::Shape> {
                [$(l[$li]),+]
            }
        }
    };
    (S1<$lp:ident> + $right:ident<$($rp:ident::$ri:literal),+> => $out:ident) => {
        impl<
            const $lp: usize,
            $(const $rp: usize),+
        > Concat<$right<$($rp),+>> for S1<$lp> {
            type Shape = $out<$lp, $($rp),+>;

            fn concat(self, _: $right<$($rp),+>) -> Self::Shape { $out }

            fn concat_indices(
                l: Self::Index,
                r: IndexOf<$right<$($rp),+>>
            ) -> IndexOf<Self::Shape>
            {
                [l, $(r[$ri]),+]
            }
        }
    };
    ($left:ident<$($lp:ident::$li:literal),+> + $right:ident<$rp:ident> => $out:ident) => {
        impl<
            $(const $lp: usize),+,
            const $rp: usize
        > Concat<S1<$rp>> for $left<$($lp),+> {
            type Shape = $out<$($lp),+, $rp>;

            fn concat(self, _: $right<$rp>) -> Self::Shape { $out }

            fn concat_indices(
                l: Self::Index,
                r: IndexOf<$right<$rp>>
            ) -> IndexOf<Self::Shape>
            {
                [$(l[$li]),+, r]
            }
        }
    };
    ($left:ident<$($lp:ident::$li:literal),+> + $right:ident<$($rp:ident::$ri:literal),+> => $out:ident) => {
        impl<
            $(const $lp: usize),+,
            $(const $rp: usize),+
        > Concat<$right<$($rp),+>> for $left<$($lp),+> {
            type Shape = $out<$($lp),+, $($rp),+>;

            fn concat(self, _: $right<$($rp),+>) -> Self::Shape { $out }

            fn concat_indices(
                l: Self::Index,
                r: IndexOf<$right<$($rp),+>>
            ) -> IndexOf<Self::Shape>
            {
                [$(l[$li]),+, $(r[$ri]),+]
            }
        }
    }
}

// S0 + ...
impl Concat<S0> for S0 {
    type Shape = S0;

    fn concat(self, _: S0) -> Self::Shape { S0 }

    fn concat_indices(_: (), _: ()) -> IndexOf<Self::Shape> { () }
}

impl<const R: usize> Concat<S1<R>> for S0 {
    type Shape = S1<R>;

    fn concat(self, _: S1<R>) -> Self::Shape { S1 }

    fn concat_indices(_: (), r: usize) -> IndexOf<Self::Shape> { r }
}

impl_concat!(S0 + S2<A1::0, B1::1>);
impl_concat!(S0 + S3<A1::0, B1::1, C1::2>);
impl_concat!(S0 + S4<A1::0, B1::1, C1::2, D1::3>);
impl_concat!(S0 + S5<A1::0, B1::1, C1::2, D1::3, E1::4>);

// S1 + ...
impl<const L: usize> Concat<S0> for S1<L> {
    type Shape = S1<L>;

    fn concat(self, _: S0) -> Self::Shape { S1 }

    fn concat_indices(l: usize, _: ()) -> IndexOf<Self::Shape> { l }
}

impl<const L: usize, const R: usize> Concat<S1<R>> for S1<L> {
    type Shape = S2<L, R>;

    fn concat(self, _: S1<R>) -> Self::Shape { S2 }

    fn concat_indices(l: usize, r: usize) -> IndexOf<Self::Shape> { [l, r] }
}

impl_concat!(S1<A1> + S2<A2::0, B2::1> => S3);
impl_concat!(S1<A1> + S3<A2::0, B2::1, C2::2> => S4);
impl_concat!(S1<A1> + S4<A2::0, B2::1, C2::2, D2::3> => S5);
impl_concat!(S1<A1> + S5<A2::0, B2::1, C2::2, D2::3, E2::4> => S6);

// S2 + ...
impl_concat!(S2<A1::0, B1::1> + S0);
impl_concat!(S2<A1::0, B1::1> + S1<A2> => S3);
impl_concat!(S2<A1::0, B1::1> + S2<A2::0, B2::1> => S4);
impl_concat!(S2<A1::0, B1::1> + S3<A2::0, B2::1, C2::2> => S5);
impl_concat!(S2<A1::0, B1::1> + S4<A2::0, B2::1, C2::2, D2::3> => S6);

// S3 + ...
impl_concat!(S3<A1::0, B1::1, C1::2> + S0);
impl_concat!(S3<A1::0, B1::1, C1::2> + S1<A2> => S4);
impl_concat!(S3<A1::0, B1::1, C1::2> + S2<A2::0, B2::1> => S5);
impl_concat!(S3<A1::0, B1::1, C1::2> + S3<A2::0, B2::1, C2::2> => S6);

// S4 + ...
impl_concat!(S4<A1::0, B1::1, C1::2, D1::3> + S0);
impl_concat!(S4<A1::0, B1::1, C1::2, D1::3> + S1<A2> => S5);
impl_concat!(S4<A1::0, B1::1, C1::2, D1::3> + S2<A2::0, B2::1> => S6);

// S5 + ...
impl_concat!(S5<A1::0, B1::1, C1::2, D1::3, E1::4> + S0);
impl_concat!(S5<A1::0, B1::1, C1::2, D1::3, E1::4> + S1<A2> => S6);

// impl_concat!(S2<A1, B1> + S2<A2, B2> => S4);
// impl_concat!(S2<A1, B1> + S3<A2, B2, C2> => S5);
// impl_concat!(S2<A1, B1> + S3<A2, B2, C2> => S5);
