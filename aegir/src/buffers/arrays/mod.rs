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

mod buffer;
mod zip_ops;
mod dot_ops;

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
