use super::*;

pub trait IntoSpec {
    type Buffer: Buffer;

    fn into_spec(self) -> Spec<Self::Buffer>;
}

impl<B: Buffer> IntoSpec for B {
    type Buffer = B;

    fn into_spec(self) -> Spec<B> { Spec::Raw(self) }
}

/// A "lifted" buffer representation.
///
/// In many cases, a given buffer instance has structural properties that allow us to prune
/// redundant compute or memory demands. For example, elementwise multiplication of two buffers
/// when either value is "all-zeroes" is certian to yield another "all-zeroes" buffer. Similarly,
/// buffer full of only one value need not be representated in a dense form. This type is designed
/// to facilitate sparse/structural representations throughout `aegir`.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub enum Spec<B: Buffer> {
    /// The raw buffer instance.
    ///
    /// This case is exactly equivalent to the original buffer.
    /// It incorporates no structural knowledge at all.
    Raw(B),

    /// A homogeneous buffer that is full of a given scalar value.
    ///
    /// In this case, we embed the fact that the buffer contains
    /// the same value in every entry. As a result, we need only store
    /// the shape of the buffer, and the value itself.
    Full(B::Shape, B::Field),

    /// A buffer with homogeneous diagonal entries only.
    ///
    /// In this case, we embed the fact that the buffer is diagonal.
    /// As a result, we need only store the shape of the buffer, and
    /// the value along the diagonal.
    Diagonal(B::Shape, B::Field),
}

impl<B: Buffer> IntoSpec for Spec<B> {
    type Buffer = B;

    fn into_spec(self) -> Spec<B> { self }
}

impl<B: Buffer> shapes::Shaped for Spec<B> {
    type Shape = B::Shape;

    fn shape(&self) -> B::Shape {
        match self {
            &Spec::Raw(ref b) => b.shape(),
            &Spec::Full(s, _) | &Spec::Diagonal(s, _) => s,
        }
    }
}

impl<F, B> Spec<B>
where
    F: Scalar,
    B: Buffer<Field = F>,
{
    /// Perform an element-wise transformation of the buffer.
    ///
    /// # Examples
    /// ```
    /// # use aegir::buffers::{Buffer, Spec, shapes::S1};
    /// let spec: Spec<[f64; 20]> = Spec::Full(S1, 0.0f64);
    /// let new_spec = spec.map(|x| x + 5.0f64);
    ///
    /// assert_eq!(new_spec, Spec::Full(S1, 5.0f64));
    /// ```
    pub fn map<A: Scalar, M: Fn(F) -> A>(
        self,
        f: M,
    ) -> Spec<<B::Class as Class<B::Shape>>::Buffer<A>> {
        match self {
            Spec::Raw(buffer) => Spec::Raw(buffer.map(f)),
            Spec::Full(shape, value) => Spec::Full(shape, f(value)),
            Spec::Diagonal(shape, value) => Spec::Raw({
                let zero = F::zero();

                <B::Class as Class<B::Shape>>::build(shape, |ix| {
                    if ix.is_diagonal() {
                        f(value)
                    } else {
                        f(zero)
                    }
                })
            }),
        }
    }

    pub fn zeroes(shape: B::Shape) -> Self { Spec::Full(shape, B::Field::zero()) }

    pub fn into_zeroes(self) -> Self {
        let zero = B::Field::zero();

        match self {
            Spec::Raw(b) => Spec::Full(b.shape(), zero),
            Spec::Full(s, _) => Spec::Full(s, zero),
            Spec::Diagonal(s, _) => Spec::Diagonal(s, zero),
        }
    }

    pub fn ones(shape: B::Shape) -> Self { Spec::Full(shape, B::Field::one()) }

    pub fn into_ones(self) -> Self {
        let one = B::Field::one();

        match self {
            Spec::Raw(b) => Spec::Full(b.shape(), one),
            Spec::Full(s, _) => Spec::Full(s, one),
            Spec::Diagonal(s, _) => Spec::Diagonal(s, one),
        }
    }

    pub fn unwrap(self) -> B {
        match self {
            Spec::Raw(b) => b,
            Spec::Full(s, x) => <B::Class as Class<B::Shape>>::full(s, x),
            Spec::Diagonal(s, x) => <B::Class as Class<B::Shape>>::diagonal(s, x),
        }
    }
}

impl<B: Buffer> From<B> for Spec<B> {
    fn from(buf: B) -> Self { Spec::Raw(buf) }
}
