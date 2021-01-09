#[macro_use]
extern crate aegir;
extern crate rand;

use aegir::{Node, Identifier, Function, Differentiable, Compile};
use ndarray::Array1;

ids!(X::x, Y::y, W::w);
state!(State {
    input: X,
    output: Y,
    weights: W
});

macro_rules! get_state {
    ($x:ident, $y:ident, $w:ident) => {
        &State { input: &$x, output: &$y, weights: &$w, }
    }
}

fn main() {
    let mut weights = Array1::from(vec![0.0; 2]);

    let x = X.to_var();
    let y = Y.to_var();
    let w = W.to_var();

    let sse = w.dot(x).sub(y).squared();

    println!("{}", sse);
    println!("{}", sse.compile_grad(W).unwrap());

    for _ in 0..10000 {
        let x_ = Array1::from(vec![rand::random::<f64>(), rand::random::<f64>()]);
        let y_ = x_[0] * 2.0 - x_[1] * 4.0;
        let g = sse.grad(get_state!(x_, y_, weights), W).unwrap();

        weights.iter_mut().zip(g.iter()).for_each(|(w, dw)| { *w -= 0.01 * dw });
    }

    println!("{:?}", weights.to_vec());
}
