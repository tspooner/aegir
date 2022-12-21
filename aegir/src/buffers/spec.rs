use super::*;

pub trait IntoSpec {
    type Buffer: Buffer;

    fn into_spec(self) -> Spec<Self::Buffer>;
}

impl<B: Buffer> IntoSpec for B {
    type Buffer = B;

    fn into_spec(self) -> Spec<B> { Spec::Raw(self) }
}

#[non_exhaustive]
#[derive(Debug, Clone)]
pub enum Spec<B: Buffer> {
    Raw(B),
    Full(B::Shape, B::Field),
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
