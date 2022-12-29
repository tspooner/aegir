use super::{
    shapes::{IndexOf, Shape, Shaped, S0, S1, S2, S3, S4, S5},
    Buffer,
    Class,
    IncompatibleShapes,
    Scalar,
    ZipFold,
    ZipMap,
};

/// Array buffer class.
pub struct Arrays;

mod buffer;
mod contraction_ops;
mod zip_ops;

#[cfg(test)]
mod tests {
    use super::*;

    mod arr2 {
        use super::*;

        const V: [f64; 2] = [1.0, 2.0];

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
