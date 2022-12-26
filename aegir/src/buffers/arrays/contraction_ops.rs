#![allow(unused, unused_mut)]
use super::*;
use crate::buffers::{Spec, Contract, shapes::ShapeOf};
use num_traits::FromPrimitive;

macro_rules! impl_contract {
    ($axes:literal; <$f:ident: $bound:ident $(, $d:ident)*> $left:ty, $right:ty; |$self:ident, $x:ident, $y:ident| -> $out:ty $c_impl:block, $s_impl:block; $shape:expr) => {
        impl<$f: $bound + Scalar $(, const $d: usize)*> Contract<$right, $axes> for $left
        {
            type Output = $out;

            fn contract(mut $self, mut $y: $right) -> Result<
                $out, IncompatibleShapes<<$left as Shaped>::Shape, <$right as Shaped>::Shape>
            >
            {
                $c_impl
            }

            fn contract_spec(mut $x: Spec<$left>, mut $y: Spec<$right>) -> Result<
                Spec<$out>, IncompatibleShapes<<$left as Shaped>::Shape, <$right as Shaped>::Shape>
            >
            {
                $s_impl
            }

            fn contract_shape($x: ShapeOf<$left>, $y: ShapeOf<$right>) -> Result<
                ShapeOf<$out>, IncompatibleShapes<<$left as Shaped>::Shape, <$right as Shaped>::Shape>
            >
            {
                Ok($shape)
            }
        }
    }
}

// Tensor product contractions:
impl_contract!(0; <F: Scalar> F, F; |self, x, y| -> F {
    Ok(self * y)
}, {
    Ok(Spec::Full(S0, x.unwrap() * y.unwrap()))
}; S0);

impl_contract!(0; <F: Scalar, N> [F; N], F; |self, x, y| -> [F; N] {
    for i in 0..N {
        self[i] = self[i] * y;
    }

    Ok(self)
}, {
    match x {
        Spec::Full(sx, fx) => Ok(Spec::Full(sx, fx * y.unwrap())),
        x => x.unwrap().contract(y.unwrap()).map(Spec::Raw),
    }
}; S1);

impl_contract!(0; <F: Scalar, N> F, [F; N]; |self, x, y| -> [F; N] {
    for i in 0..N {
        y[i] = y[i] * self;
    }

    Ok(y)
}, {
    match y {
        Spec::Full(sy, fy) => Ok(Spec::Full(sy, x.unwrap() * fy)),
        y => x.unwrap().contract(y.unwrap()).map(Spec::Raw),
    }
}; S1);

impl_contract!(0; <F: Scalar, N, M> [[F; M]; N], F; |self, x, y| -> [[F; M]; N] {
    for i in 0..N {
        for j in 0..M {
            self[i][j] = self[i][j] * y;
        }
    }

    Ok(self)
}, {
    match x {
        Spec::Full(sx, fx) => Ok(Spec::Full(sx, fx * y.unwrap())),
        x => x.unwrap().contract(y.unwrap()).map(Spec::Raw),
    }
}; S2);

impl_contract!(0; <F: Scalar, N, M> [F; N], [F; M]; |self, x, y| -> [[F; M]; N] {
    Ok(self.map(|x_i| y.map(|y_i| x_i * y_i)))
}, {
    use Spec::*;

    match (x, y) {
        (Full(_, fx), Full(_, fy)) => Ok(Full(S2, fx * fy)),

        (Full(_, fx), _) if fx.is_zero() => Ok(Spec::zeroes(S2)),
        (_, Full(_, fy)) if fy.is_zero() => Ok(Spec::zeroes(S2)),

        (Full(_, fx), y) => {
            let y = y.unwrap();

            Ok(Raw(array_init::array_init(|_| y.map(|y_i| fx * y_i))))
        },

        (x, Full(_, fy)) => {
            let x = x.unwrap();

            Ok(Raw(x.map(|x_i| array_init::array_init(|_| x_i * fy))))
        },
        (x, y) => x.unwrap().contract(y.unwrap()).map(Raw),
    }
}; S2);

impl_contract!(0; <F: Scalar, N, M> F, [[F; M]; N]; |self, x, y| -> [[F; M]; N] {
    for i in 0..N {
        for j in 0..M {
            y[i][j] = y[i][j] * self;
        }
    }

    Ok(y)
}, {
    match y {
        Spec::Full(sy, fy) => Ok(Spec::Full(sy, fy * x.unwrap())),
        y => x.unwrap().contract(y.unwrap()).map(Spec::Raw),
    }
}; S2);

// Tensor dot product contractions:
impl_contract!(1; <F: Scalar> F, F; |self, x, y| -> F {
    Ok(self * y)
}, {
    Ok(Spec::Full(S0, x.unwrap() * y.unwrap()))
}; S0);

impl_contract!(1; <F: FromPrimitive, N> [F; N], [F; N]; |self, x, y| -> F {
    self.zip_fold(&y, num_traits::zero(), |acc, (xi, yi)| acc + xi * yi)
}, {
    use Spec::*;

    match (x, y) {
        (Full(_, fx), Full(_, fy)) => Ok(Full(S0, F::from_usize(N).unwrap() * fx * fy)),
        (Full(_, fx), y) => Ok(Full(S0, y.unwrap().fold(F::zero(), |acc, yi| acc + fx * yi))),
        (x, Full(_, fy)) => Ok(Full(S0, x.unwrap().fold(F::zero(), |acc, xi| acc + xi * fy))),
        (x, y) => x.unwrap().zip_fold(&y.unwrap(), F::zero(), |acc, (xi, yi)| acc + xi * yi).map(Raw),
    }
}; S0);

impl_contract!(1; <F: Scalar, N, M> [[F; M]; N], [F; N]; |self, x, y| -> [F; M] {
    Ok(array_init::array_init(|i| (0..M).fold(F::zero(), |acc, k| acc + self[i][k] * y[k])))
}, {
    use Spec::*;

    match (x, y) {
        (Full(_, fx), _) if fx.is_zero() => Ok(Spec::zeroes(S1)),

        (Diagonal(_, fx), Full(_, fy)) => Ok(Full(S1, fx * fy)),

        (Diagonal(_, fx), y) if M == N => Ok(Raw({
            let mut y = y.unwrap();
            y.mutate(|yi| fx * yi);

            let ptr = &mut y as *mut _ as *mut [F; M];
            let res = unsafe { ptr.read() };

            core::mem::forget(y);

            res
        })),

        (x, y) => x.unwrap().contract(y.unwrap()).map(Spec::Raw),
    }
}; S1);

impl_contract!(1; <F: FromPrimitive, N, M, Z> [[F; Z]; N], [[F; M]; Z]; |self, x, y| -> [[F; M]; N] {
    Ok(array_init::array_init(|i| {
        array_init::array_init(|j| {
            (0..Z).fold(F::zero(), |acc, z| acc + self[i][z] * y[z][j])
        })
    }))
}, {
    use Spec::*;

    match (x, y) {
        (Full(_, fx), Full(_, fy)) => Ok(Full(S2, F::from_usize(Z).unwrap() * fx * fy)),

        (Diagonal(_, fx), Diagonal(_, fy)) if N == Z && M == N => Ok(Diagonal(S2, fx * fy)),

        (x, y) => x.unwrap().contract(y.unwrap()).map(Spec::Raw),
    }
}; S2);

impl_contract!(1; <F: Scalar, N, M> [F; N], [[F; M]; N]; |self, x, y| -> [F; M] {
    Ok(array_init::array_init(|i| (0..M).fold(F::zero(), |acc, k| acc + self[k] * y[k][i])))
}, {
    use Spec::*;

    match (x, y) {
        (_, Full(_, fy)) if fy.is_zero() => Ok(Spec::zeroes(S1)),

        (Full(_, fx), Diagonal(_, fy)) => Ok(Full(S1, fx * fy)),

        (x, Diagonal(_, fy)) if M == N => Ok(Raw({
            let mut x = x.unwrap();
            x.mutate(|xi| xi * fy);

            let ptr = &mut x as *mut _ as *mut [F; M];
            let res = unsafe { ptr.read() };

            core::mem::forget(x);

            res
        })),

        (x, y) => x.unwrap().contract(y.unwrap()).map(Spec::Raw),
    }
}; S1);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_r0_r0() {
        let x = 2.0f64;
        let y = 4.0f64;

        assert_eq!(Contract::<_, 1>::contract(x, y).unwrap(), 8.0);
    }
    #[test]
    fn test_r1_r2() {
        let x = [-1.0f64, 1.0f64];
        let y = [[1.0f64, 2.0f64], [3.0f64, 4.0f64]];

        assert_eq!(Contract::<_, 1>::contract(x, y).unwrap(), [2.0, 2.0]);
    }

    #[test]
    fn test_r2_r1() {
        let x = [[1.0f64, 2.0f64], [3.0f64, 4.0f64]];
        let y = [-1.0f64, 1.0f64];

        assert_eq!(Contract::<_, 1>::contract(x, y).unwrap(), [1.0, 1.0]);
    }

    #[test]
    fn test_r2_r2() {
        let x = [[1.0f64, 2.0f64], [3.0f64, 4.0f64]];

        assert_eq!(
            Contract::<_, 1>::contract(x, x).unwrap(),
            [[7.0, 10.0], [15.0, 22.0]]
        );
    }
}
