use super::*;
use crate::buffers::Contract;

macro_rules! impl_contract {
    ($axes:literal; <$f:ident $(, $d:ident)*> $left:ty, $right:ty; |$x:ident, $y:ident| -> $out:ty { $impl:expr }, $shape:expr) => {
        impl<$f: Scalar $(, const $d: usize)*> Contract<$right, $axes> for $left
        {
            type Output = $out;

            fn contract($x, $y: $right) -> Result<
                $out, IncompatibleShapes<<$left as Buffer>::Shape, <$right as Buffer>::Shape>
            >
            {
                $impl
            }

            fn contract_shape(_: <$left as Buffer>::Shape, _: <$right as Buffer>::Shape) -> Result<
                <$out as Buffer>::Shape, IncompatibleShapes<<$left as Buffer>::Shape, <$right as Buffer>::Shape>
            >
            {
                Ok($shape)
            }
        }
    }
}

// Tensor product contractions:
impl_contract!(0; <F> F, F; |self, y| -> F { Ok(self * y) }, S0);

impl_contract!(0; <F, N> [F; N], F; |self, y| -> [F; N] { Ok(self.map(|xi| xi * y)) }, S1);
impl_contract!(0; <F, N> F, [F; N]; |self, y| -> [F; N] { Ok(y.map(|yi| self * yi)) }, S1);

impl_contract!(0; <F, N, M> [[F; M]; N], F; |self, y| -> [[F; M]; N] {
    Ok(self.map(|x_i| x_i.map(|x_ij| x_ij * y)))
}, S2);
impl_contract!(0; <F, N, M> [F; N], [F; M]; |self, y| -> [[F; M]; N] {
    Ok(self.map(|x_i| y.map(|y_i| x_i * y_i)))
}, S2);
impl_contract!(0; <F, N, M> F, [[F; M]; N]; |self, y| -> [[F; M]; N] {
    Ok(y.map(|y_i| y_i.map(|y_ij| self * y_ij)))
}, S2);

// Tensor dot product contractions:
impl_contract!(1; <F> F, F; |self, y| -> F { Ok(self * y) }, S0);

impl_contract!(1; <F, N> [F; N], [F; N]; |self, y| -> F {
    self.zip_fold(&y, num_traits::zero(), |acc, (xi, yi)| acc + xi * yi)
}, S0);

impl_contract!(1; <F, N, M> [[F; M]; N], [F; N]; |self, y| -> [F; M] {
    Ok(array_init::array_init(|i| (0..M).fold(F::zero(), |acc, k| acc + self[i][k] * y[k])))
}, S1);
impl_contract!(1; <F, N, M, Z> [[F; Z]; N], [[F; M]; Z]; |self, y| -> [[F; M]; N] {
    Ok(array_init::array_init(|i| {
        array_init::array_init(|j| {
            (0..Z).fold(F::zero(), |acc, z| acc + self[i][z] * y[z][j])
        })
    }))
}, S2);
impl_contract!(1; <F, N, M> [F; N], [[F; M]; N]; |self, y| -> [F; M] {
    Ok(array_init::array_init(|i| (0..M).fold(F::zero(), |acc, k| acc + self[k] * y[k][i])))
}, S1);

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
