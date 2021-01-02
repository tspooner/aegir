#[macro_use]
extern crate aegir;
extern crate rand;

use aegir::{Get, GetError, Differentiable};
use ndarray::Array1;

ids!(W::w, X::x, Y::y);

impl std::cmp::PartialEq<W> for X { fn eq(&self, _: &W) -> bool { false } }
impl std::cmp::PartialEq<W> for Y { fn eq(&self, _: &W) -> bool { false } }

impl std::cmp::PartialEq<X> for Y { fn eq(&self, _: &X) -> bool { false } }
impl std::cmp::PartialEq<X> for W { fn eq(&self, _: &X) -> bool { false } }

impl std::cmp::PartialEq<Y> for X { fn eq(&self, _: &Y) -> bool { false } }
impl std::cmp::PartialEq<Y> for W { fn eq(&self, _: &Y) -> bool { false } }

pub struct State<'a> {
    pub input: &'a Array1<f64>,
    pub output: &'a Array1<f64>,
    pub weights: &'a Array1<f64>,
}

impl<'a> Get<X> for State<'a> {
    type Output = Array1<f64>;

    fn get(&self, _: X) -> Result<&Array1<f64>, GetError<X>> {
        Ok(self.input)
    }
}

impl<'a> Get<Y> for State<'a> {
    type Output = Array1<f64>;

    fn get(&self, _: Y) -> Result<&Array1<f64>, GetError<Y>> {
        Ok(self.output)
    }
}

impl<'a> Get<W> for State<'a> {
    type Output = Array1<f64>;

    fn get(&self, _: W) -> Result<&Array1<f64>, GetError<W>> {
        Ok(self.weights)
    }
}

fn main() {
    let mut weights = Array1::from(vec![0.0; 1]);

    let x = aegir::sources::Variable(X);
    let y = aegir::sources::Variable(Y);

    let model = aegir::ops::scalar::Mul(aegir::sources::Variable(W), x);
    let error = aegir::ops::scalar::Add(y, model.clone());

    let sq_error = aegir::ops::scalar::Power(error, 2.0);
    let sse = aegir::ops::reductions::Sum(sq_error);

    for _ in 0..10000 {
        let x_ = Array1::from(vec![rand::random::<f64>()]);
        let y_ = x_.clone() * 2.0;

        let d = sse.dual(W, &State {
            input: &x_,
            output: &y_,
            weights: &weights,
        }).unwrap();

        weights.iter_mut().zip(d.adjoint.iter()).for_each(|(w, dw)| {
            *w -= 0.01 * dw
        });
    }

    println!("{:?}", weights.to_vec());
}
