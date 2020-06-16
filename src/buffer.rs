use num_traits::{Float, Zero};

/// Gradient buffer with arbitrary dimension.
pub trait Buffer: Clone {
    type Elem: Float;

    fn id(elem: Self::Elem) -> Self;
}

pub trait BufferMut: Buffer {
    fn reset(&mut self) { self.fill(Self::Elem::zero())}

    fn fill(&mut self, value: Self::Elem) { self.map_inplace(|_| value) }

    fn map(&self, f: impl Fn(Self::Elem) -> Self::Elem) -> Self {
        self.clone().map_into(f)
    }

    fn map_into(mut self, f: impl Fn(Self::Elem) -> Self::Elem) -> Self {
        self.map_inplace(f);
        self
    }

    fn map_inplace(&mut self, f: impl Fn(Self::Elem) -> Self::Elem);

    fn merge(&self, other: &Self, f: impl Fn(Self::Elem, Self::Elem) -> Self::Elem) -> Self {
        self.clone().merge_into(other, f)
    }

    fn merge_into(mut self, other: &Self, f: impl Fn(Self::Elem, Self::Elem) -> Self::Elem) -> Self {
        self.merge_inplace(other, f);
        self
    }

    fn merge_inplace(&mut self, other: &Self, f: impl Fn(Self::Elem, Self::Elem) -> Self::Elem);
}

pub trait IntoBuffer {
    type IntoBuffer: Buffer;

    fn into_buffer(self) -> Self::IntoBuffer;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Implement Buffer traits for f64
///////////////////////////////////////////////////////////////////////////////////////////////////
impl Buffer for f64 {
    type Elem = f64;

    fn id(elem: f64) -> Self { elem }
}

impl BufferMut for f64 {
    fn fill(&mut self, value: f64) { *self = value; }

    fn map(&self, f: impl Fn(f64) -> f64) -> Self { f(*self) }

    fn map_into(self, f: impl Fn(f64) -> f64) -> Self { f(self) }

    fn map_inplace(&mut self, f: impl Fn(f64) -> f64) { *self = f(*self) }

    fn merge(&self, other: &Self, f: impl Fn(f64, f64) -> f64) -> Self {
        f(*self, *other)
    }

    fn merge_into(self, other: &Self, f: impl Fn(f64, f64) -> f64) -> Self {
        f(self, *other)
    }

    fn merge_inplace(&mut self, other: &Self, f: impl Fn(f64, f64) -> f64) {
        *self = f(*self, *other)
    }
}

impl<T: Buffer, U: Buffer<Elem = T::Elem>> Buffer for (T, U) {
    type Elem = T::Elem;

    fn id(elem: T::Elem) -> (T, U) { (T::id(elem.clone()), U::id(elem)) }
}

impl<T: BufferMut, U: BufferMut<Elem = T::Elem>> BufferMut for (T, U) {
    fn fill(&mut self, value: T::Elem) {
        self.0.fill(value);
        self.1.fill(value);
    }

    fn map(&self, f: impl Fn(T::Elem) -> T::Elem) -> Self {
        (self.0.map(&f), self.1.map(f))
    }

    fn map_into(self, f: impl Fn(T::Elem) -> T::Elem) -> Self {
        (self.0.map_into(&f), self.1.map_into(f))
    }

    fn map_inplace(&mut self, f: impl Fn(T::Elem) -> T::Elem) {
        self.0.map_inplace(&f);
        self.1.map_inplace(f);
    }

    fn merge(&self, other: &Self, f: impl Fn(T::Elem, T::Elem) -> T::Elem) -> Self {
        (self.0.merge(&other.0, &f), self.1.merge(&other.1, f))
    }

    fn merge_into(self, other: &Self, f: impl Fn(T::Elem, T::Elem) -> T::Elem) -> Self {
        (self.0.merge_into(&other.0, &f), self.1.merge_into(&other.1, f))
    }

    fn merge_inplace(&mut self, other: &Self, f: impl Fn(T::Elem, T::Elem) -> T::Elem) {
        self.0.merge_inplace(&other.0, &f);
        self.1.merge_inplace(&other.1, f);
    }
}

impl<T: Buffer> Buffer for [T; 2] {
    type Elem = T::Elem;

    fn id(elem: T::Elem) -> [T; 2] { [T::id(elem), T::id(elem)] }
}

impl<T: BufferMut> BufferMut for [T; 2] {
    fn fill(&mut self, value: T::Elem) {
        self[0].fill(value);
        self[1].fill(value);
    }

    fn map(&self, f: impl Fn(T::Elem) -> T::Elem) -> Self {
        [self[0].map(&f), self[1].map(f)]
    }

    fn map_into(self, f: impl Fn(T::Elem) -> T::Elem) -> Self {
        let [x, y] = self;

        [x.map_into(&f), y.map_into(f)]
    }

    fn map_inplace(&mut self, f: impl Fn(T::Elem) -> T::Elem) {
        self[0].map_inplace(&f);
        self[1].map_inplace(f);
    }

    fn merge(&self, other: &Self, f: impl Fn(T::Elem, T::Elem) -> T::Elem) -> Self {
        [self[0].merge(&other[0], &f), self[1].merge(&other[1], f)]
    }

    fn merge_into(self, other: &Self, f: impl Fn(T::Elem, T::Elem) -> T::Elem) -> Self {
        let [x, y] = self;

        [x.merge_into(&other[0], &f), y.merge_into(&other[1], f)]
    }

    fn merge_inplace(&mut self, other: &Self, f: impl Fn(T::Elem, T::Elem) -> T::Elem) {
        self[0].merge_inplace(&other[0], &f);
        self[1].merge_inplace(&other[1], f);
    }
}
