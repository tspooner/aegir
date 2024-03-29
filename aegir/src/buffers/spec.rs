use super::*;

/// Trait for types that can be converted into a [Spec].
///
/// The main purpose of this trait is to maintain a clear separation between buffers and specs,
/// where the latter is a "lifted" variant of the former. Since it is automatically implemented
/// for `B: Buffer` and `Spec`, we can always be certain that any `T: IntoSpec` is either a buffer
/// or a spec itself. Therefore, we can use this trait to generalise across "Buffer-Like" types in
/// in our abstractions.
pub trait IntoSpec {
    /// The corresponding buffer type underlying the `Spec`.
    type Buffer: Buffer;

    /// Convert `self` into a [Spec] instance.
    ///
    /// # Examples
    /// ```
    /// # use aegir::buffers::{Spec, IntoSpec};
    /// let spec = [1, 2, 3, 4, 5].into_spec();
    ///
    /// assert_eq!(spec, Spec::Raw([1, 2, 3, 4, 5]));
    /// ```
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
/// when either value is "all-zeroes" is certian to yield another "all-zeroes" buffer. Similarly, a
/// buffer full of only one value need not be representated in a dense form. This type is designed
/// to facilitate such sparse/structural representations throughout `aegir`.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub enum Spec<B: Buffer> {
    // TODO - think of ways to reduce memory overhead. The whole purpose of this type is to avoid
    // redundant compute, but we still allocate memory on the stack for these intermediate results,
    // even if they're never materialised.
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
    /// Returns the contained buffer in dense form, consuming `self`.
    ///
    /// This method is not necessarily cheap to execute as it may require a call to [Class::build].
    #[inline]
    pub fn unwrap(self) -> B {
        match self {
            Spec::Raw(b) => b,
            Spec::Full(s, x) => <B::Class as Class<B::Shape>>::full(s, x),
            Spec::Diagonal(s, x) => <B::Class as Class<B::Shape>>::diagonal(s, x),
        }
    }

    /// Apply `Buffer::map` operation in lifted `Spec` context.
    ///
    /// # Examples
    ///
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

    /// Apply `ZipMap::zip_map` operation in lifted `Spec` context.
    ///
    /// # Examples
    ///
    /// ```
    /// # use aegir::buffers::{Buffer, Spec, shapes::S1};
    /// let a: Spec<[f64; 10]> = Spec::Full(S1, 0.0f64);
    /// let b: Spec<[f64; 10]> = Spec::Full(S1, 1.0f64);
    ///
    /// let new_spec = a.zip_map(&b, |x, y| x + y).unwrap();
    ///
    /// assert_eq!(new_spec, Spec::Full(S1, 1.0f64));
    /// ```
    pub fn zip_map<R, T, M: Fn(B::Field, R::Field) -> T>(
        self,
        other: &Spec<R>,
        f: M,
    ) -> Result<Spec<B::Output<T>>, IncompatibleShapes<B::Shape, R::Shape>>
    where
        B: ZipMap<R>,
        B::Shape: Broadcast<R::Shape>,

        R: Buffer,
        T: Scalar,
    {
        use Spec::*;

        match (self, other) {
            (Full(sx, fx), &Full(sy, fy)) => sx.broadcast(sy).map(|sz| Full(sz, f(fx, fy))),

            // TODO XXX - THis is painfully inefficient right now. Improve!
            (lhs, rhs) => lhs.unwrap().zip_map(&rhs.clone().unwrap(), f).map(Spec::Raw),
        }
    }

    /// Construct a `Spec::Full(s, 0)` where s is a given shape.
    #[inline]
    pub fn zeroes(shape: B::Shape) -> Self { Spec::Full(shape, B::Field::zero()) }

    /// Construct a `Spec::Full(s, 1)` where s is a given shape.
    #[inline]
    pub fn ones(shape: B::Shape) -> Self { Spec::Full(shape, B::Field::one()) }

    /// Construct a `Spec::Diagonal(s, 1)` where s is a given shape.
    #[inline]
    pub fn eye(shape: B::Shape) -> Self { Spec::Diagonal(shape, B::Field::one()) }
}

impl<B: Buffer> From<B> for Spec<B> {
    fn from(buf: B) -> Self { Spec::Raw(buf) }
}
