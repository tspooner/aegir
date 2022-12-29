use crate::buffers::{ZipMap, ZipMut, ZipFold};
use super::*;

macro_rules! impl_mut {
    (
        @Type <$f:ident, $($d:ident),+> $arr:ty;

        @zm_self |$zms_self:ident, $zms_rhs:ident, $zms_func:ident| $zms_impl:block;
        @zm_field_r |$zmfr_self:ident, $zmfr_rhs:ident, $zmfr_func:ident| $zmfr_impl:block;
    ) => {
        impl<$f: Scalar, $(const $d: usize),+> ZipMut for $arr {
            #[inline]
            fn zip_mut<M: Fn($f, $f) -> $f>(
                &mut $zms_self,
                $zms_rhs: &Self,
                $zms_func: M,
            ) -> Result<(), IncompatibleShapes<Self::Shape>> {
                $zms_impl
            }
        }

        impl<$f: Scalar, $(const $d: usize),+> ZipMut<$f> for $arr {
            #[inline]
            fn zip_mut<M: Fn($f, $f) -> $f>(
                &mut $zmfr_self,
                $zmfr_rhs: &$f,
                $zmfr_func: M,
            ) -> Result<(), IncompatibleShapes<Self::Shape, S0>> {
                $zmfr_impl
            }
        }
    };
    (<$f:ident, $($d:ident),+> $arr:ty) => {
        impl_mut!(
            @Type <$f, $($d),+> $arr;

            @zm_self |self, rhs, f| {
                for i in 0..self.len() {
                    self[i].zip_mut(&rhs[i], &f).ok();
                }

                Ok(())
            };
            @zm_field_r |self, rhs, f| {
                for i in 0..self.len() {
                    self[i].zip_mut(rhs, &f).ok();
                }

                Ok(())
            };
        );
    }
}

macro_rules! impl_map {
    (
        @Type <$f:ident, $a:ident, $($d:ident),+> $arr:ty, $garr:ty;

        @zm_self |$zms_self:ident, $zms_rhs:ident, $zms_func:ident| $zms_impl:block;
        @zm_field_r |$zmfr_self:ident, $zmfr_rhs:ident, $zmfr_func:ident| $zmfr_impl:block;
        @zm_field_l |$zmfl_self:ident, $zmfl_rhs:ident, $zmfl_func:ident| $zmfl_impl:block;
    ) => {
        impl<$f: Scalar, $(const $d: usize),+> ZipMap for $arr {
            type Output<A: Scalar> = $garr;

            #[inline]
            fn zip_map<$a: Scalar, M: Fn($f, $f) -> $a>(
                $zms_self,
                $zms_rhs: &Self,
                $zms_func: M,
            ) -> Result<$garr, IncompatibleShapes<Self::Shape>> {
                $zms_impl
            }

            #[inline]
            fn zip_map_id<M: Fn($f, $f) -> $f>(
                mut self,
                rhs: &Self,
                f: M,
            ) -> Result<$arr, IncompatibleShapes<Self::Shape>> {
                self.zip_mut(rhs, f).map(|_| self)
            }

            #[inline]
            fn zip_shape(
                self,
                _: Self::Shape,
            ) -> Result<$arr, IncompatibleShapes<Self::Shape>> {
                Ok(self)
            }
        }

        impl<$f: Scalar, $(const $d: usize),+> ZipMap<$f> for $arr {
            type Output<A: Scalar> = $garr;

            #[inline]
            fn zip_map<$a: Scalar, M: Fn($f, $f) -> $a>(
                $zmfr_self,
                $zmfr_rhs: &$f,
                $zmfr_func: M,
            ) -> Result<$garr, IncompatibleShapes<Self::Shape, S0>> {
                $zmfr_impl
            }

            #[inline]
            fn zip_map_id<M: Fn($f, $f) -> $f>(
                mut self,
                rhs: &$f,
                f: M,
            ) -> Result<$arr, IncompatibleShapes<Self::Shape, S0>> {
                self.zip_mut(rhs, f).map(|_| self)
            }

            #[inline]
            fn zip_shape(
                self,
                _: S0,
            ) -> Result<$arr, IncompatibleShapes<Self::Shape, S0>> {
                Ok(self)
            }
        }

        impl<$f: Scalar, $(const $d: usize),+> ZipMap<$arr> for $f {
            type Output<A: Scalar> = $garr;

            #[inline]
            fn zip_map<$a: Scalar, M: Fn($f, $f) -> $a>(
                $zmfl_self,
                $zmfl_rhs: &$arr,
                $zmfl_func: M,
            ) -> Result<$garr, IncompatibleShapes<S0, <$arr as Shaped>::Shape>> {
                $zmfl_impl
            }

            #[inline]
            fn zip_shape(
                self,
                rshape: <$arr as Shaped>::Shape,
            ) -> Result<$arr, IncompatibleShapes<S0, <$arr as Shaped>::Shape>> {
                Ok(Arrays::full(rshape, self))
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
                    self[i].zip_map(rhs, &f).unwrap_unchecked()
                }))
            };
            @zm_field_l |self, rhs, f| {
                Ok(array_init::array_init(|i| unsafe {
                    rhs[i].zip_map(&self, &f).unwrap_unchecked()
                }))
            };
        );
    }
}

macro_rules! impl_fold {
    (<$f:ident, $a:ident, $($d:ident),+> $arr:ty) => {
        impl<$f: Scalar, $(const $d: usize),+> ZipFold for $arr {
            #[inline]
            fn zip_fold<$a: Scalar, M: Fn($a, ($f, $f)) -> $a>(
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
            #[inline]
            fn zip_fold<$a: Scalar, M: Fn($a, ($f, $f)) -> $a>(
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
impl_mut!(
    @Type <F, D1> [F; D1];

    @zm_self |self, rhs, f| {
        for i in 0..D1 {
            self[i] = f(self[i], rhs[i]);
        }

        Ok(())
    };
    @zm_field_r |self, rhs, f| {
        for i in 0..D1 {
            self[i] = f(self[i], *rhs);
        }

        Ok(())
    };
);
impl_map!(
    @Type <F, A, D1> [F; D1], [A; D1];

    @zm_self |self, rhs, f| {
        Ok(array_init::array_init(|i| f(self[i], rhs[i])))
    };
    @zm_field_r |self, rhs, f| {
        Ok(array_init::array_init(|i| f(self[i], *rhs)))
    };
    @zm_field_l |self, rhs, f| {
        Ok(array_init::array_init(|i| f(self, rhs[i])))
    };
);
impl_fold!(<F, A, D1> [F; D1]);

// Rank-2 Tensor:
impl_mut!(<F, D1, D2> [[F; D2]; D1]);
impl_map!(<F, A, D1, D2> [[F; D2]; D1], [[A; D2]; D1]);
impl_fold!(<F, A, D1, D2> [[F; D2]; D1]);

// Rank-3 Tensor:
impl_mut!(<F, D1, D2, D3> [[[F; D3]; D2]; D1]);
impl_map!(<F, A, D1, D2, D3> [[[F; D3]; D2]; D1], [[[A; D3]; D2]; D1]);
impl_fold!(<F, A, D1, D2, D3> [[[F; D3]; D2]; D1]);

// Rank-4 Tensor:
impl_mut!(<F, D1, D2, D3, D4> [[[[F; D4]; D3]; D2]; D1]);
impl_map!(<F, A, D1, D2, D3, D4> [[[[F; D4]; D3]; D2]; D1], [[[[A; D4]; D3]; D2]; D1]);
impl_fold!(<F, A, D1, D2, D3, D4> [[[[F; D4]; D3]; D2]; D1]);

// Rank-5 Tensor:
impl_mut!(<F, D1, D2, D3, D4, D5> [[[[[F; D5]; D4]; D3]; D2]; D1]);
impl_map!(<F, A, D1, D2, D3, D4, D5> [[[[[F; D5]; D4]; D3]; D2]; D1], [[[[[A; D5]; D4]; D3]; D2]; D1]);
impl_fold!(<F, A, D1, D2, D3, D4, D5> [[[[[F; D5]; D4]; D3]; D2]; D1]);
