use num_traits::{NumOps, Zero, One};

/// Trait for numeric types implementing basic scalar operations.
pub trait Field: Copy + Buffer<Field = Self> + NumOps + std::fmt::Debug {}

impl<T> Field for T
where
    T: Copy + Buffer<Field = Self> + NumOps + std::fmt::Debug,
{}

/// Trait for types defining a vector space over a [Field](Field).
pub trait Buffer: std::fmt::Debug {
    type Field: Field;
    type Owned: OwnedBuffer<Field = Self::Field>;

    /// Creates an [Owned](Buffer::Owned) instance from a borrowed buffer, usually by cloning.
    fn to_owned(&self) -> Self::Owned;

    /// Convert buffer directly into an [Owned](Buffer::Owned) instance, cloning when necessary.
    fn into_owned(self) -> Self::Owned;

    fn to_zeroes(&self) -> Self::Owned where Self::Field: num_traits::Zero {
        self.to_filled(num_traits::identities::zero())
    }

    fn into_zeroes(self) -> Self::Owned where Self: Sized, Self::Field: num_traits::Zero {
        self.into_filled(num_traits::identities::zero())
    }

    fn to_ones(&self) -> Self::Owned where Self::Field: num_traits::One {
        self.to_filled(num_traits::identities::one())
    }

    fn into_ones(self) -> Self::Owned where Self: Sized, Self::Field: num_traits::One {
        self.into_filled(num_traits::identities::one())
    }

    fn to_filled(&self, value: Self::Field) -> Self::Owned {
        self.to_owned().map(|_| value)
    }

    fn into_filled(self, value: Self::Field) -> Self::Owned where Self: Sized {
        self.map(|_| value)
    }

    /// Perform an elementwise transformation of the buffer.
    ///
    /// # Examples
    /// ```
    /// # use aegir::buffer::Buffer;
    /// let buffer = vec![0.0, 1.0, 2.0, 3.0];
    /// let new_buffer = buffer.map(|el| el * 2.0);
    ///
    /// assert_eq!(new_buffer[0], 0.0);
    /// assert_eq!(new_buffer[1], 2.0);
    /// assert_eq!(new_buffer[2], 4.0);
    /// assert_eq!(new_buffer[3], 6.0);
    /// ```
    fn map<F>(self, f: F) -> Self::Owned
    where F: Fn(Self::Field) -> Self::Field;

    /// Perform a fold over the elements of the buffer.
    ///
    /// # Examples
    /// ```
    /// # use aegir::buffer::Buffer;
    /// let buffer = vec![0.0, 1.0, 2.0, 3.0];
    ///
    /// assert_eq!(buffer.fold(0.0, |init, &el| init + 2.0 * el), 12.0);
    /// ```
    fn fold<F>(&self, init: Self::Field, f: F) -> Self::Field
    where F: Fn(Self::Field, &Self::Field) -> Self::Field;

    /// Sum over the elements of the buffer.
    ///
    /// # Examples
    /// ```
    /// # use aegir::buffer::Buffer;
    /// let buffer = vec![0.0, 1.0, 2.0, 3.0];
    ///
    /// assert_eq!(buffer.sum(), 6.0);
    /// assert_eq!(buffer.map(|el| 2.0 * el).sum(), 12.0);
    /// ```
    fn sum(&self) -> Self::Field where Self::Field: num_traits::Zero {
        self.fold(num_traits::zero(), |init, &el| init + el)
    }
}

/// Trait for owned [Buffer] types.
pub trait OwnedBuffer: Buffer<Owned = Self> + Clone {}

impl<B: Buffer<Owned = B> + Clone> OwnedBuffer for B {}

/// Type shortcut for the [Field] associated with a [Buffer].
pub type FieldOf<B> = <B as Buffer>::Field;

/// Type shortcut for the [Owned](Buffer::Owned) variant of a [Buffer].
pub type OwnedOf<B> = <B as Buffer>::Owned;

impl Buffer for f64 {
    type Field = f64;
    type Owned = f64;

    fn to_owned(&self) -> f64 { *self }

    fn into_owned(self) -> f64 { self }

    fn map<F: Fn(f64) -> Self::Field>(self, f: F) -> f64 { f(self) }

    fn fold<F: Fn(f64, &f64) -> f64>(&self, init: f64, f: F) -> f64 { f(init, self) }
}

impl Buffer for &f64 {
    type Field = f64;
    type Owned = f64;

    fn to_owned(&self) -> f64 { **self }

    fn into_owned(self) -> f64 { *self }

    fn map<F: Fn(f64) -> Self::Field>(self, f: F) -> f64 { f(*self) }

    fn fold<F: Fn(f64, &f64) -> f64>(&self, init: f64, f: F) -> f64 { f(init, self) }
}

impl<F: Field> Buffer for Vec<F> {
    type Field = F;
    type Owned = Self;

    fn to_owned(&self) -> Vec<F> { self.clone() }

    fn into_owned(self) -> Vec<F> { self }

    fn map<Func: Fn(F) -> F>(self, f: Func) -> Self {
        self.into_iter().map(f).collect()
    }

    fn fold<Func: Fn(F, &F) -> F>(&self, init: F, f: Func) -> Self::Field {
        self.into_iter().fold(init, f)
    }
}

impl<F: Field> Buffer for &Vec<F> {
    type Field = F;
    type Owned = Vec<F>;

    fn to_owned(&self) -> Vec<F> { self.to_vec() }

    fn into_owned(self) -> Vec<F> { self.to_vec() }

    fn map<Func: Fn(F) -> F>(self, f: Func) -> Vec<F> {
        self.into_iter().map(|x| f(*x)).collect()
    }

    fn fold<Func: Fn(F, &F) -> F>(&self, init: F, f: Func) -> Self::Field {
        self.into_iter().fold(init, f)
    }
}

impl<F, S> Buffer for ndarray::ArrayBase<S, ndarray::Ix1>
where
    F: Field,
    S: ndarray::Data<Elem = F> + ndarray::RawDataClone,
{
    type Field = F;
    type Owned = ndarray::Array1<F>;

    fn to_owned(&self) -> ndarray::Array1<F> { self.to_owned() }

    fn into_owned(self) -> ndarray::Array1<F> { self.into_owned() }

    fn map<Func: Fn(F) -> F>(self, f: Func) -> ndarray::Array1<F> {
        self.into_owned().mapv(f)
    }

    fn fold<Func: Fn(F, &F) -> F>(&self, init: F, f: Func) -> F {
        self.fold(init, f)
    }
}

impl<F, S> Buffer for &ndarray::ArrayBase<S, ndarray::Ix1>
where
    F: Field,
    S: ndarray::Data<Elem = F> + ndarray::RawDataClone,
{
    type Field = F;
    type Owned = ndarray::Array1<F>;

    fn to_owned(&self) -> ndarray::Array1<F> { (*self).to_owned() }

    fn into_owned(self) -> ndarray::Array1<F> { ndarray::ArrayBase::into_owned(self.clone()) }

    fn map<Func: Fn(F) -> F>(self, f: Func) -> ndarray::Array1<F> {
        self.into_owned().mapv(f)
    }

    fn fold<Func: Fn(F, &F) -> F>(&self, init: F, f: Func) -> F {
        ndarray::ArrayBase::fold(self, init, f)
    }
}

impl<F, S> Buffer for ndarray::ArrayBase<S, ndarray::Ix2>
where
    F: Field,
    S: ndarray::Data<Elem = F> + ndarray::RawDataClone,
{
    type Field = F;
    type Owned = ndarray::Array2<F>;

    fn to_owned(&self) -> ndarray::Array2<F> { self.to_owned() }

    fn into_owned(self) -> ndarray::Array2<F> { self.into_owned() }

    fn map<Func: Fn(F) -> F>(self, f: Func) -> ndarray::Array2<F> {
        self.into_owned().mapv(f)
    }

    fn fold<Func: Fn(F, &F) -> F>(&self, init: F, f: Func) -> F {
        self.fold(init, f)
    }
}

impl<F, S> Buffer for &ndarray::ArrayBase<S, ndarray::Ix2>
where
    F: Field,
    S: ndarray::Data<Elem = F> + ndarray::RawDataClone,
{
    type Field = F;
    type Owned = ndarray::Array2<F>;

    fn to_owned(&self) -> ndarray::Array2<F> { (*self).to_owned() }

    fn into_owned(self) -> ndarray::Array2<F> { ndarray::ArrayBase::into_owned(self.clone()) }

    fn map<Func: Fn(F) -> F>(self, f: Func) -> ndarray::Array2<F> {
        self.into_owned().mapv(f)
    }

    fn fold<Func: Fn(F, &F) -> F>(&self, init: F, f: Func) -> F {
        ndarray::ArrayBase::fold(self, init, f)
    }
}
