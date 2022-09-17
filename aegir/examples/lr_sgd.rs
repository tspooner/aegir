#[macro_use]
extern crate aegir;
extern crate rand;

use aegir::{
    ids::{W, X, Y},
    Differentiable,
    Function,
    Identifier,
    Node,
};

db!(Database { x: X, y: Y, w: W });

fn main() {
    let mut weights = [0.0, 0.0];

    let x = X.into_var();
    let y = Y.into_var();
    let w = W.into_var();

    let model = x.dot(w);

    // Using standard method calls...
    let sse = model.sub(y).squared();
    let adj = sse.adjoint(W);

    // ...or using aegir! macro
    let sse = aegir!((model - y) ^ 2);
    let adj = sse.adjoint(W);

    for _ in 0..100000 {
        let [x1, x2] = [rand::random::<f64>(), rand::random::<f64>()];

        let g = adj
            .evaluate(Database {
                // Independent variables:
                x: [x1, x2],

                // Dependent variable:
                y: x1 * 2.0 - x2 * 4.0,

                // Model weights:
                w: &weights,
            })
            .unwrap();

        weights[0] -= 0.01 * g[0][0];
        weights[1] -= 0.01 * g[0][1];
    }

    println!("{:?}", weights.to_vec());
}
