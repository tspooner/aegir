#[macro_use]
extern crate aegir;
extern crate rand;

use aegir::{
    ids::{W, X, Y},
    ops,
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

    let adj = ops::TensorMul(
        ops::Double(ops::Sub(ops::InnerProduct(x, w), y)),
        ops::TensorMul(w.adjoint(W), x)
    );

    for _ in 0..1_000_000 {
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
