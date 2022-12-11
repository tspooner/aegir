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
    let mut db = Database {
        x: [0.0, 0.0],
        y: 0.0,
        w: [0.0, 0.0],
    };

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
        // Independent variables:
        db.x = rng.gen();

        // Dependent variable:
        db.y = db.x[0] * 2.0 - db.x[1] * 4.0;

        // Evaluate gradient:
        let g = adj.evaluate(&db).unwrap();

        weights[0] -= 0.01 * g[0];
        weights[1] -= 0.01 * g[1];
    }

    println!("{:?}", weights.to_vec());
}
