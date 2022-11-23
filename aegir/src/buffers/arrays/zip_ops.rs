use super::*;

macro_rules! impl_map {
    (
        @Type <$f:ident, $a:ident, $($d:ident),+> $arr:ty, $garr:ty;

        @zm_self |$zms_self:ident, $zms_rhs:ident, $zms_func:ident| { $zms_impl:expr };
        @zm_field_r |$zmfr_self:ident, $zmfr_rhs:ident, $zmfr_func:ident| { $zmfr_impl:expr };
        @zm_field_l |$zmfl_self:ident, $zmfl_rhs:ident, $zmfl_func:ident| { $zmfl_impl:expr };
    ) => {
        impl<$f: Scalar, $(const $d: usize),+> ZipMap for $arr {
            type Output<$a: Scalar> = $garr;

            fn zip_map<$a: Scalar, M: Fn(F, F) -> $a>(
                &$zms_self,
                $zms_rhs: &Self,
                $zms_func: M,
            ) -> Result<$garr, IncompatibleShapes<Self::Shape>> {
                $zms_impl
            }

            fn zip_map_left<$a: Scalar, M: Fn(F) -> $a>(
                &self,
                _: Self::Shape,
                f: M,
            ) -> Result<$garr, IncompatibleShapes<Self::Shape>> {
                Ok(self.map(f))
            }

            fn zip_map_right<$a: Scalar, M: Fn(F) -> $a>(
                _: Self::Shape,
                rhs: &Self,
                f: M,
            ) -> Result<$garr, IncompatibleShapes<Self::Shape>> {
                Ok(rhs.map(f))
            }

            fn zip_map_neither<$a: Scalar>(
                shape: Self::Shape,
                _: Self::Shape,
                fill_value: $a,
            ) -> Result<$garr, IncompatibleShapes<Self::Shape>> {
                Ok(Arrays::full(shape, fill_value))
            }
        }

        impl<$f: Scalar, $(const $d: usize),+> ZipMap<$f> for $arr {
            type Output<$a: Scalar> = $garr;

            fn zip_map<$a: Scalar, M: Fn(F, F) -> $a>(
                &$zmfr_self,
                $zmfr_rhs: &$f,
                $zmfr_func: M,
            ) -> Result<$garr, IncompatibleShapes<Self::Shape, S0>> {
                $zmfr_impl
            }

            fn zip_map_left<$a: Scalar, M: Fn(F) -> $a>(
                &self,
                _: S0,
                f: M,
            ) -> Result<$garr, IncompatibleShapes<Self::Shape, S0>> {
                Ok(self.map(f))
            }

            fn zip_map_right<$a: Scalar, M: Fn(F) -> $a>(
                lhs_shape: Self::Shape,
                rhs: &$f,
                f: M,
            ) -> Result<$garr, IncompatibleShapes<Self::Shape, S0>> {
                Ok(Arrays::full(lhs_shape, f(*rhs)))
            }

            fn zip_map_neither<$a: Scalar>(
                shape: Self::Shape,
                _: S0,
                fill_value: $a,
            ) -> Result<$garr, IncompatibleShapes<Self::Shape, S0>> {
                Ok(Arrays::full(shape, fill_value))
            }
        }

        impl<$f: Scalar, $(const $d: usize),+> ZipMap<$arr> for $f {
            type Output<$a: Scalar> = $garr;

            fn zip_map<$a: Scalar, M: Fn(F, F) -> $a>(
                &$zmfl_self,
                $zmfl_rhs: &$arr,
                $zmfl_func: M,
            ) -> Result<$garr, IncompatibleShapes<S0, <$arr as Buffer>::Shape>> {
                $zmfl_impl
            }

            fn zip_map_left<$a: Scalar, M: Fn(F) -> $a>(
                &self,
                rhs_shape: <$arr as Buffer>::Shape,
                f: M,
            ) -> Result<$garr, IncompatibleShapes<S0, <$arr as Buffer>::Shape>> {
                Ok(Arrays::full(rhs_shape, f(*self)))
            }

            fn zip_map_right<$a: Scalar, M: Fn(F) -> $a>(
                _: S0,
                rhs: &$arr,
                f: M,
            ) -> Result<$garr, IncompatibleShapes<S0, <$arr as Buffer>::Shape>> {
                Ok(rhs.map(f))
            }

            fn zip_map_neither<$a: Scalar>(
                _: Self::Shape,
                shape: <$arr as Buffer>::Shape,
                fill_value: $a,
            ) -> Result<$garr, IncompatibleShapes<S0, <$arr as Buffer>::Shape>> {
                Ok(Arrays::full(shape, fill_value))
            }
        }
    };
    (<$f:ident, $a:ident, $($d:ident),+> $arr:ty, $garr:ty) => {
        impl_map!(
            @Type <$f, $a, $($d),+> $arr, $garr;

            @zm_self |self, rhs, f| {
                Ok(array_init::array_init(|i| unsafe {
                    self[i].zip_map(&rhs[i], &f).unwrap_unchecked()
                }))
            };
            @zm_field_r |self, rhs, f| {
                Ok(array_init::array_init(|i| unsafe {
                    self[i].zip_map(rhs, |x, y| f(x, y)).unwrap_unchecked()
                }))
            };
            @zm_field_l |self, rhs, f| {
                Ok(array_init::array_init(|i| unsafe {
                    rhs[i].zip_map(self, |x, y| f(y, x)).unwrap_unchecked()
                }))
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
impl_map!(
    @Type <F, A, D1> [F; D1], [A; D1];

    @zm_self |self, rhs, f| { Ok(array_init::array_init(|i| f(self[i], rhs[i]))) };
    @zm_field_r |self, rhs, f| { Ok(array_init::array_init(|i| f(self[i], *rhs))) };
    @zm_field_l |self, rhs, f| { Ok(array_init::array_init(|i| f(*self, rhs[i]))) };
);

impl_fold!(<F, A, D1> [F; D1]);

// Rank-2 Tensor:
impl_map!(<F, A, D1, D2> [[F; D2]; D1], [[A; D2]; D1]);
impl_fold!(<F, A, D1, D2> [[F; D2]; D1]);

// Rank-3 Tensor:
impl_map!(<F, A, D1, D2, D3> [[[F; D3]; D2]; D1], [[[A; D3]; D2]; D1]);
impl_fold!(<F, A, D1, D2, D3> [[[F; D3]; D2]; D1]);

// Rank-4 Tensor:
impl_map!(<F, A, D1, D2, D3, D4> [[[[F; D4]; D3]; D2]; D1], [[[[A; D4]; D3]; D2]; D1]);
impl_fold!(<F, A, D1, D2, D3, D4> [[[[F; D4]; D3]; D2]; D1]);

// Rank-5 Tensor:
impl_map!(<F, A, D1, D2, D3, D4, D5> [[[[[F; D5]; D4]; D3]; D2]; D1], [[[[[A; D5]; D4]; D3]; D2]; D1]);
impl_fold!(<F, A, D1, D2, D3, D4, D5> [[[[[F; D5]; D4]; D3]; D2]; D1]);
