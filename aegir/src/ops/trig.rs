use num_traits::real::Real;
use std::ops::Neg;

impl_real!(@unary Cos["cos"], |x| { x.cos() }, |x| { -x.sin() });
impl_real!(@unary Cosh["cosh"], |x| { x.cosh() }, |x| { x.sinh() });
impl_real!(@unary ArcCos["acos"], |x| { x.acos() }, |x| {
    (x.powi(2) - num_traits::one()).neg().sqrt().recip().neg()
});
impl_real!(@unary ArcCosh["acosh"], |x| { x.acosh() }, |x| {
    (x.powi(2) - num_traits::one()).sqrt().recip()
});

impl_real!(@unary Sin["sin"], |x| { x.sin() }, |x| { x.cos() });
impl_real!(@unary Sinh["sinh"], |x| { x.sinh() }, |x| { x.cosh() });
impl_real!(@unary ArcSin["asin"], |x| { x.asin() }, |x| {
    (x.powi(2) - num_traits::one()).neg().sqrt().recip()
});
impl_real!(@unary ArcSinh["asinh"], |x| { x.asinh() }, |x| {
    (x.powi(2) + num_traits::one()).sqrt().recip()
});

impl_real!(@unary Tan["tan"], |x| { x.tan() }, |x| { x.cos().powi(2).recip() });
impl_real!(@unary Tanh["tanh"], |x| { x.tanh() }, |x| { x.cosh().powi(2).recip() });
impl_real!(@unary ArcTan["atan"], |x| { x.atan() }, |x| {
    (x.powi(2) + num_traits::one()).recip()
});
impl_real!(@unary ArcTanh["atanh"], |x| { x.atanh() }, |x| {
    (x.powi(2) - num_traits::one()).neg().recip()
});
