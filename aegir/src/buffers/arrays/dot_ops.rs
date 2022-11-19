use crate::buffers::Contract;
use super::*;

macro_rules! impl_tdot {
    (@Array <$f:ident $(, $d:ident)*> $left:ty, $right:ty; |$x:ident, $y:ident| -> $out:ty { $impl:expr }) => {
        impl<$f: Scalar $(, const $d: usize)*> Contract<$right, 1> for $left
        {
            type Output = $out;

            fn contract($x, $y: $right) -> Result<
                $out, IncompatibleShapes<<$left as Buffer>::Shape, <$right as Buffer>::Shape>
            >
            {
                $impl
            }
        }
    }
}

impl_tdot!(@Array <F> F, F; |self, y| -> F { Ok(self * y) });

impl_tdot!(@Array <F, N> F, [F; N]; |self, y| -> [F; N] { Ok(y.map(|yi| self * yi)) });
impl_tdot!(@Array <F, N> [F; N], F; |self, y| -> [F; N] { Ok(self.map(|xi| xi * y)) });
impl_tdot!(@Array <F, N> [F; N], [F; N]; |self, y| -> F {
    self.zip_fold(&y, num_traits::zero(), |acc, (xi, yi)| acc + xi * yi)
});

impl_tdot!(@Array <F, N, M> [[F; M]; N], [F; N]; |self, y| -> [F; M] {
    Ok(array_init::array_init(|i| (0..M).fold(F::zero(), |acc, k| acc + self[i][k] * y[k])))
});
impl_tdot!(@Array <F, N, M, Z> [[F; Z]; N], [[F; M]; Z]; |self, y| -> [[F; M]; N] {
    Ok(array_init::array_init(|i| {
        array_init::array_init(|j| {
            (0..Z).fold(F::zero(), |acc, z| acc + self[i][z] * y[z][j])
        })
    }))
});
impl_tdot!(@Array <F, N, M> [F; N], [[F; M]; N]; |self, y| -> [F; M] {
    Ok(array_init::array_init(|i| (0..M).fold(F::zero(), |acc, k| acc + self[k] * y[k][i])))
});


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_r0_r0() {
        let x = 2.0f64;
        let y = 4.0f64;

        assert_eq!(x.contract(y).unwrap(), 8.0);
    }
    #[test]
    fn test_r1_r2() {
        let x = [-1.0f64, 1.0f64];
        let y = [[1.0f64, 2.0f64], [3.0f64, 4.0f64]];

        assert_eq!(x.contract(y).unwrap(), [2.0, 2.0]);
    }

    #[test]
    fn test_r2_r1() {
        let x = [[1.0f64, 2.0f64], [3.0f64, 4.0f64]];
        let y = [-1.0f64, 1.0f64];

        assert_eq!(x.contract(y).unwrap(), [1.0, 1.0]);
    }

    #[test]
    fn test_r2_r2() {
        let x = [[1.0f64, 2.0f64], [3.0f64, 4.0f64]];

        assert_eq!(x.contract(x).unwrap(), [
            [7.0, 10.0],
            [15.0, 22.0]
        ]);
    }
}
