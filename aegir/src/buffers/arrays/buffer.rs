use super::*;

macro_rules! impl_buffer {
    (
        @Type <$f:ident, $($d:ident),+>($shape:ty) $arr:ty;

        @build |$b_func:ident| { $b_impl:expr };
        @build_subset |$bs_buf:ident, $bs_ix:ident, $bs_func:ident| { $bs_impl:expr };
        @full |$f_val:ident| { $f_impl:expr };

        @shape { $s_impl:expr };
        @get_unchecked |$gu_self:ident, $gu_ix:ident| { $gu_impl:expr };
        @map |$m_self:ident, $m_func:ident| { $m_impl:expr };
    ) => {
        // Class implementation:
        impl<$(const $d: usize),+> Class<$shape> for Arrays {
            type Buffer<$f: Scalar> = $arr;

            fn build<F: Scalar>(_: $shape, $b_func: impl Fn(IndexOf<$shape>) -> F) -> Self::Buffer<F> {
                $b_impl
            }

            fn build_subset<F: Scalar>(
                shape: $shape,
                base: F,
                indices: impl Iterator<Item = IndexOf<$shape>>,
                $bs_func: impl Fn(IndexOf<$shape>) -> F,
            ) -> Self::Buffer<F> {
                let mut $bs_buf = Self::full(shape, base);

                for $bs_ix in indices { $bs_impl }

                $bs_buf
            }

            fn full<F: Scalar>(_: $shape, $f_val: F) -> Self::Buffer<F> { $f_impl }
        }

        impl<$f: Scalar, $(const $d: usize),+> Shaped for $arr {
            type Shape = $shape;

            fn shape(&self) -> Self::Shape { $s_impl }
        }

        impl<$f: Scalar, $(const $d: usize),+> Buffer for $arr {
            type Class = Arrays;
            type Field = $f;

            fn class() -> Arrays { Arrays }

            fn get_unchecked(&$gu_self, $gu_ix: IndexOf<Self::Shape>) -> $f { $gu_impl }

            fn map<A: Scalar, M: Fn($f) -> A>($m_self, $m_func: M) -> <Arrays as Class<$shape>>::Buffer<A> { $m_impl }

            fn map_ref<A: Scalar, M: Fn($f) -> A>(&self, f: M) -> <Arrays as Class<$shape>>::Buffer<A> { array_init::array_init(|i| self[i].map_ref(&f)) }

            fn fold<A, M: Fn(A, $f) -> A>(&self, mut init: A, f: M) -> A {
                for i in 0..D1 {
                    init = self[i].fold(init, &f)
                }

                init
            }
        }
    };
}

// Rank-1 Tensor:
impl_buffer!(
    @Type <F, D1>(S1<D1>) [F; D1];

    @build |f| { array_init::array_init(f) };
    @build_subset |buf, ix, f| { buf[ix] = f(ix) };
    @full |val| { [val; D1] };

    @shape { S1 };
    @get_unchecked |self, ix| { self[ix] };
    @map |self, f| { <[F; D1]>::map(self, f) };
);

// Rank-2 Tensor:
impl_buffer!(
    @Type <F, D1, D2>(S2<D1, D2>) [[F; D2]; D1];

    @build |f| { array_init::array_init(|i| array_init::array_init(|j| f([i, j]))) };
    @build_subset |buf, ix, f| { buf[ix[0]][ix[1]] = f(ix) };
    @full |val| { [[val; D2]; D1] };

    @shape { S2 };
    @get_unchecked |self, ix| { self[ix[0]][ix[1]] };
    @map |self, f| { <[[F; D2]; D1]>::map(self, |x| Buffer::map(x, &f)) };
);

// Rank-3 Tensor:
impl_buffer!(
    @Type <F, D1, D2, D3>(S3<D1, D2, D3>) [[[F; D3]; D2]; D1];

    @build |f| {
        array_init::array_init(|i| {
            array_init::array_init(|j| {
                array_init::array_init(|k| f([i, j, k]))
            })
        })
    };
    @build_subset |buf, ix, f| { buf[ix[0]][ix[1]][ix[2]] = f(ix) };
    @full |val| { [[[val; D3]; D2]; D1] };

    @shape { S3 };
    @get_unchecked |self, ix| { self[ix[0]][ix[1]][ix[2]] };
    @map |self, f| { <[[[F; D3]; D2]; D1]>::map(self, |x| Buffer::map(x, &f)) };
);

// Rank-4 Tensor:
impl_buffer!(
    @Type <F, D1, D2, D3, D4>(S4<D1, D2, D3, D4>) [[[[F; D4]; D3]; D2]; D1];

    @build |f| {
        array_init::array_init(|i| {
            array_init::array_init(|j| {
                array_init::array_init(|k| {
                    array_init::array_init(|l| f([i, j, k, l]))
                })
            })
        })
    };
    @build_subset |buf, ix, f| { buf[ix[0]][ix[1]][ix[2]][ix[3]] = f(ix) };
    @full |val| { [[[[val; D4]; D3]; D2]; D1] };

    @shape { S4 };
    @get_unchecked |self, ix| { self[ix[0]][ix[1]][ix[2]][ix[3]] };
    @map |self, f| { <[[[[F; D4]; D3]; D2]; D1]>::map(self, |x| Buffer::map(x, &f)) };
);

// Rank-5 Tensor:
impl_buffer!(
    @Type <F, D1, D2, D3, D4, D5>(S5<D1, D2, D3, D4, D5>) [[[[[F; D5]; D4]; D3]; D2]; D1];

    @build |f| {
        array_init::array_init(|i| {
            array_init::array_init(|j| {
                array_init::array_init(|k| {
                    array_init::array_init(|l| {
                        array_init::array_init(|m| f([i, j, k, l, m]))
                    })
                })
            })
        })
    };
    @build_subset |buf, ix, f| { buf[ix[0]][ix[1]][ix[2]][ix[3]][ix[4]] = f(ix) };
    @full |val| { [[[[[val; D5]; D4]; D3]; D2]; D1] };

    @shape { S5 };
    @get_unchecked |self, ix| { self[ix[0]][ix[1]][ix[2]][ix[3]][ix[4]] };
    @map |self, f| { <[[[[[F; D5]; D4]; D3]; D2]; D1]>::map(self, |x| Buffer::map(x, &f)) };
);
