use super::{
    shapes::{IndexOf, Shape, S0, S1, S2, S3, S4, S5},
    Buffer,
    Class,
    ZipMap,
    IncompatibleShapes,
    Scalar,
    ZipFold,
};

/// Array buffer class.
pub struct Arrays;

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

        // Buffer (owned) implementation:
        impl<$f: Scalar, $(const $d: usize),+> Buffer for $arr {
            type Class = Arrays;
            type Field = $f;
            type Shape = $shape;

            fn shape(&self) -> Self::Shape { $s_impl }

            fn get_unchecked(&$gu_self, $gu_ix: IndexOf<Self::Shape>) -> $f { $gu_impl }

            fn map<A: Scalar, M: Fn($f) -> A>($m_self, $m_func: M) -> <Arrays as Class<$shape>>::Buffer<A> { $m_impl }

            fn map_ref<A: Scalar, M: Fn($f) -> A>(&self, f: M) -> <Arrays as Class<$shape>>::Buffer<A> { array_init::array_init(|i| self[i].map_ref(&f)) }

            fn fold<A, M: Fn(A, $f) -> A>(&self, mut init: A, f: M) -> A {
                for i in 0..D1 {
                    init = self[i].fold(init, &f)
                }

                init
            }

            fn to_owned(&self) -> <Arrays as Class<$shape>>::Buffer<$f> { self.clone() }

            fn into_owned(self) -> <Arrays as Class<$shape>>::Buffer<$f> { self }
        }

        // Buffer (ref) implementation:
        impl<$f: Scalar, $(const $d: usize),+> Buffer for &$arr {
            type Class = Arrays;
            type Field = $f;
            type Shape = $shape;

            fn shape(&self) -> Self::Shape { $s_impl }

            fn map<A: Scalar, M: Fn($f) -> A>(self, f: M) -> <Arrays as Class<$shape>>::Buffer<A> {
                <$arr as Buffer>::map_ref(self, f)
            }

            fn get_unchecked(&$gu_self, $gu_ix: IndexOf<Self::Shape>) -> $f { $gu_impl }

            fn map_ref<A: Scalar, M: Fn($f) -> A>(&self, f: M) -> <Arrays as Class<$shape>>::Buffer<A> { array_init::array_init(|i| self[i].map_ref(&f)) }

            fn fold<A, M: Fn(A, $f) -> A>(&self, mut init: A, f: M) -> A {
                for i in 0..D1 {
                    init = self[i].fold(init, &f)
                }

                init
            }

            fn to_owned(&self) -> <Arrays as Class<$shape>>::Buffer<$f> { <$arr as Buffer>::to_owned(self) }

            fn into_owned(self) -> <Arrays as Class<$shape>>::Buffer<$f> { self.clone() }
        }
    };
}

macro_rules! impl_zip {
    (
        @Type <$f:ident, $a:ident, $($d:ident),+> $arr:ty, $garr:ty;

        @zm_self |$zms_self:ident, $zms_rhs:ident, $zms_func:ident| { $zms_impl:expr };
        @zm_field |$zmf_self:ident, $zmf_rhs:ident, $zmf_func:ident| { $zmf_impl:expr };
    ) => {
        impl<$f: Scalar, $(const $d: usize),+> ZipMap for $arr {
            type Output<$a: Scalar> = $garr;

            fn zip_map_ref<$a: Scalar, M: Fn(F, F) -> $a>(
                &$zms_self,
                $zms_rhs: &Self,
                $zms_func: M,
            ) -> Result<$garr, IncompatibleShapes<Self::Shape>> {
                $zms_impl
            }
        }

        impl<$f: Scalar, $(const $d: usize),+> ZipMap<$f> for $arr {
            type Output<$a: Scalar> = $garr;

            fn zip_map_ref<$a: Scalar, M: Fn(F, F) -> $a>(
                &$zmf_self,
                $zmf_rhs: &$f,
                $zmf_func: M,
            ) -> Result<$garr, IncompatibleShapes<Self::Shape, S0>> {
                $zmf_impl
            }
        }
    };
    (<$f:ident, $a:ident, $($d:ident),+> $arr:ty, $garr:ty) => {
        impl_zip!(
            @Type <$f, $a, $($d),+> $arr, $garr;

            @zm_self |self, rhs, f| {
                Ok(array_init::array_init(|i| unsafe {
                    self[i].zip_map(&rhs[i], &f).unwrap_unchecked()
                }))
            };
            @zm_field |self, rhs, f| {
                Ok(array_init::array_init(|i| unsafe { self[i].zip_map(rhs, &f).unwrap_unchecked() }))
            };
        );
    }
}

macro_rules! impl_fold {
    (<$f:ident, $a:ident, $($d:ident),+> $arr:ty) => {
        impl<$f: Scalar, $(const $d: usize),+> ZipFold for $arr {
            fn zip_fold<$a: Scalar, M: Fn($a, (F, F)) -> $a>(
                &self,
                rhs: &Self,
                mut acc: $a,
                f: M,
            ) -> Result<$a, IncompatibleShapes<Self::Shape>> {
                for ijk in self.shape().indices() {
                    acc = f(acc, (self.get_unchecked(ijk), rhs.get_unchecked(ijk)))
                }

                Ok(acc)
            }
        }

        impl<$f: Scalar, $(const $d: usize),+> ZipFold<$f> for $arr {
            fn zip_fold<$a: Scalar, M: Fn($a, (F, F)) -> $a>(
                &self,
                rhs: &$f,
                mut acc: $a,
                f: M,
            ) -> Result<$a, IncompatibleShapes<Self::Shape, S0>> {
                for ijk in self.shape().indices() {
                    acc = f(acc, (self.get_unchecked(ijk), *rhs))
                }

                Ok(acc)
            }
        }
    }
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

impl_zip!(
    @Type <F, A, D1> [F; D1], [A; D1];

    @zm_self |self, rhs, f| { Ok(array_init::array_init(|i| f(self[i], rhs[i]))) };
    @zm_field |self, rhs, f| { Ok(array_init::array_init(|i| f(self[i], *rhs))) };
);

impl_fold!(<F, A, D1> [F; D1]);

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

impl_zip!(<F, A, D1, D2> [[F; D2]; D1], [[A; D2]; D1]);
impl_fold!(<F, A, D1, D2> [[F; D2]; D1]);

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

impl_zip!(<F, A, D1, D2, D3> [[[F; D3]; D2]; D1], [[[A; D3]; D2]; D1]);
impl_fold!(<F, A, D1, D2, D3> [[[F; D3]; D2]; D1]);

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

impl_zip!(<F, A, D1, D2, D3, D4> [[[[F; D4]; D3]; D2]; D1], [[[[A; D4]; D3]; D2]; D1]);
impl_fold!(<F, A, D1, D2, D3, D4> [[[[F; D4]; D3]; D2]; D1]);

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

impl_zip!(<F, A, D1, D2, D3, D4, D5> [[[[[F; D5]; D4]; D3]; D2]; D1], [[[[[A; D5]; D4]; D3]; D2]; D1]);
impl_fold!(<F, A, D1, D2, D3, D4, D5> [[[[[F; D5]; D4]; D3]; D2]; D1]);

#[cfg(test)]
mod tests {
    use super::*;

    mod arr2 {
        use super::*;

        const V: [f64; 2] = [1.0, 2.0];

        #[test]
        fn test_ownership() {
            assert_eq!(Buffer::to_owned(&V), V);
            assert_eq!(Buffer::into_owned(V), V);
        }

        // #[test]
        // fn test_replace() {
            // assert_eq!(V.to_zeroes(), [0.0; 2]);
            // assert_eq!(V.into_zeroes(), [0.0; 2]);

            // assert_eq!(V.to_ones(), [1.0; 2]);
            // assert_eq!(V.into_ones(), [1.0; 2]);

            // assert_eq!(V.to_filled(5.0), [5.0; 2]);
            // assert_eq!(V.to_filled(-1.0), [-1.0; 2]);

            // assert_eq!(V.into_filled(5.0), [5.0; 2]);
            // assert_eq!(V.into_filled(-1.0), [-1.0; 2]);
        // }

        #[test]
        fn test_transforms() {
            assert_eq!(V.map(|x| x * 2.0), [2.0, 4.0]);
            assert_eq!(V.fold(0.0, |a, x| a + x * 2.0), 6.0);
            assert_eq!(V.sum(), 3.0);
        }

        #[test]
        fn test_linalg() {
            assert_eq!(
                V.zip_fold(&V, 0.0, |acc, (xi, yi)| acc + xi * yi).unwrap(),
                5.0
            );
        }
    }
}
