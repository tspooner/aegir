// use crate::buffer::Buffer;
// use num_traits::{Zero, One};
// use std::ops;

#[derive(Clone, Copy, Debug)]
pub struct Dual<V, A> {
    pub value: V,
    pub adjoint: A,
}

// impl<V: Buffer, A: Buffer> Dual<V, A> {
    // pub fn constant(value: V) -> Dual<V, A>
    // where A::Elem: Zero,
    // {
        // Dual {
            // value,
            // adjoint: A::zeros(),
        // }
    // }

    // pub fn variable(value: V) -> Dual<V, A>
    // where A::Elem: One,
    // {
        // Dual {
            // value,
            // adjoint: A::ones(),
        // }
    // }
// }

impl<V, A> Dual<V, A> {
    #[inline]
    pub fn map<V_, E_>(self, f: impl Fn(V, A) -> (V_, E_)) -> Dual<V_, E_> {
        f(self.value, self.adjoint).into()
    }

    // #[inline]
    // pub fn map_elem(self, f: impl Fn(V::Elem) -> V::Elem, g: impl Fn(A::Elem) -> A::Elem) -> Dual<V, A> {
        // self.map_val_elem(f).map_eps_elem(g)
    // }

    // #[inline]
    // pub fn map_val<V_: Buffer>(self, f: impl Fn(V) -> V_) -> Dual<V_, A> {
        // Dual {
            // value: f(self.value),
            // adjoint: self.adjoint,
        // }
    // }

    // #[inline]
    // pub fn map_val_elem(self, f: impl Fn(V::Elem) -> V::Elem) -> Dual<V, A> {
        // Dual {
            // value: self.value.map_into(f),
            // adjoint: self.adjoint,
        // }
    // }

    // #[inline]
    // pub fn map_eps<E_: Buffer>(self, f: impl Fn(A) -> E_) -> Dual<V, E_> {
        // Dual {
            // value: self.value,
            // adjoint: f(self.adjoint),
        // }
    // }

    // #[inline]
    // pub fn map_eps_elem(self, f: impl Fn(A::Elem) -> A::Elem) -> Dual<V, A> {
        // Dual {
            // value: self.value,
            // adjoint: self.adjoint.map_into(f),
        // }
    // }
}

// impl<V: Buffer, A: Buffer> Dual<V, A> {
    // pub fn conj(self) -> Self {
        // self.map_eps_elem(|x| -x)
    // }
// }

// impl<V: Buffer, A: Buffer> Dual<V, A> {
    // pub fn sin(self) -> Self {
        // self.map_elem(|x| x.sin(), |x| x.cos())
    // }

    // pub fn cos(self) -> Self {
        // self.map_elem(|x| x.cos(), |x| -x.sin())
    // }

    // pub fn tan(self) -> Self {
        // self.map_elem(|x| x.tan(), |x| {
            // let sec_x = x.cos().recip();

            // sec_x * sec_x
        // })
    // }

    // pub fn sec(self) -> Self {
        // self.map_elem(|x| x.cos().recip(), |x| x.tan() / x.cos())
    // }

    // pub fn csc(self) -> Self {
        // self.map_elem(|x| x.sin().recip(), |x| -x.sin().recip() / x.tan())
    // }

    // pub fn cot(self) -> Self {
        // self.map_elem(|x| x.tan().recip(), |x| {
            // let csc_x = -x.sin().recip();

            // csc_x * csc_x
        // })
    // }
// }

// impl<V: Buffer, A: Buffer> ops::Neg for Dual<V, A>
// {
    // type Output = Dual<V, A>;

    // fn neg(self) -> Dual<V, A> {
        // self.map_val_elem(|x| -x).map_eps_elem(|x| -x)
    // }
// }

// impl<V: Buffer, A: Buffer> ops::Neg for &Dual<V, A>
// {
    // type Output = Dual<V, A>;

    // fn neg(self) -> Dual<V, A> {
        // self.clone().map_val_elem(|x| -x).map_eps_elem(|x| -x)
    // }
// }

// impl<V: Buffer, A: Buffer> ops::Add<Dual<V, A>> for Dual<V, A>
// {
    // type Output = Dual<V, A>;

    // fn add(self, rhs: Dual<V, A>) -> Dual<V, A> {
        // self
            // .map_val(|v| v.merge_into(&rhs.value, |x, y| x + y))
            // .map_eps(|e| e.merge_into(&rhs.adjoint, |x, y| x + y))
    // }
// }

// impl<V: Buffer, A: Buffer> ops::Add<&Dual<V, A>> for Dual<V, A>
// {
    // type Output = Dual<V, A>;

    // fn add(self, rhs: &Dual<V, A>) -> Dual<V, A> {
        // self
            // .map_val(|v| v.merge_into(&rhs.value, |x, y| x + y))
            // .map_eps(|e| e.merge_into(&rhs.adjoint, |x, y| x + y))
    // }
// }

// impl<V: Buffer, A: Buffer> ops::Sub<Dual<V, A>> for Dual<V, A>
// {
    // type Output = Dual<V, A>;

    // fn sub(self, rhs: Dual<V, A>) -> Dual<V, A> {
        // self
            // .map_val(|v| v.merge_into(&rhs.value, |x, y| x - y))
            // .map_eps(|e| e.merge_into(&rhs.adjoint, |x, y| x - y))
    // }
// }

// impl<V: Buffer, A: Buffer> ops::Sub<&Dual<V, A>> for Dual<V, A>
// {
    // type Output = Dual<V, A>;

    // fn sub(self, rhs: &Dual<V, A>) -> Dual<V, A> {
        // self
            // .map_val(|v| v.merge_into(&rhs.value, |x, y| x - y))
            // .map_eps(|e| e.merge_into(&rhs.adjoint, |x, y| x - y))
    // }
// }

impl<V, A> From<(V, A)> for Dual<V, A> {
    #[inline]
    fn from((value, adjoint): (V, A)) -> Dual<V, A> {
        Dual { value, adjoint, }
    }
}
