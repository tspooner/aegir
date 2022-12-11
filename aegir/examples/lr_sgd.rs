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
use rand::{Rng, SeedableRng, rngs::SmallRng};

db!(Database { x: X, y: Y, w: W });

fn main() {
    let mut rng = SmallRng::seed_from_u64(1994);
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

    for _ in 0..1_000_000 {
        let xs: [f64; 2] = rng.gen();

        let g = adj
            .evaluate(Database {
                // Independent variables:
                x: xs,

                // Dependent variable:
                y: xs[0] * 2.0 - xs[1] * 4.0,

                // Model weights:
                w: &weights,
            })
            .unwrap();

        weights[0] -= 0.01 * g[0];
        weights[1] -= 0.01 * g[1];
    }

    println!("{:?}", weights.to_vec());
}
