#[macro_use]
extern crate aegir;
extern crate rand;

use aegir::{Compile, Differentiable, Function, Identifier, Node};
use ndarray::Array1;

ids!(X::x, Y::y, W::w);
db!(Database {
    input: X,
    output: Y,
    weights: W
});

macro_rules! make_db {
    ($x:ident, $y:ident, $w:ident) => {
        &Database {
            input: &$x,
            output: &$y,
            weights: &$w,
        }
    };
}

fn main() {
    let mut weights = Array1::from(vec![0.0; 2]);

    let x = X.to_var();
    let y = Y.to_var();
    let w = W.to_var();

    // Using standard method calls...
    let sse = w.dot(x).sub(y).squared();

    // ...or using compile! macro
    let sse = compile!((w.dot(x) - y) ^ 2.0);

    println!("{}", sse);
    println!("{}", sse.compile_grad(W).unwrap());

    for _ in 0..10000 {
        let x_ = Array1::from(vec![rand::random::<f64>(), rand::random::<f64>()]);
        let y_ = x_[0] * 2.0 - x_[1] * 4.0;
        let g = sse.grad(make_db!(x_, y_, weights), W).unwrap();

        weights
            .iter_mut()
            .zip(g.iter())
            .for_each(|(w, dw)| *w -= 0.01 * dw);
    }

    println!("{:?}", weights.to_vec());
}
