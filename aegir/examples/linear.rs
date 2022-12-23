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
use rand::{rngs::SmallRng, Rng, SeedableRng};

const N: usize = 30;

db!(Database { x: X, y: Y, w: W });

fn main() {
    let mut rng = SmallRng::seed_from_u64(1994);
    let mut db = Database {
        x: [1.0; N],
        y: 0.0,
        w: [0.0; N],
    };

    let true_weights: [f64; N] = rng.gen();

    let x = X.into_var();
    let y = Y.into_var();
    let w = W.into_var();

    let model = x.dot(w);
    let sse = model.sub(y).squared();
    let adj = sse.adjoint(W);

    for _ in 0..10_000_000 {
        // Independent variables:
        db.x = rng.gen();

        // Dependent variable:
        db.y = db.x.iter().zip(true_weights.iter()).map(|(x, tw)| x * tw).sum();

        // Evaluate gradient:
        let g: [f64; N] = adj.evaluate(&db).unwrap();

        for i in 0..N {
            db.w[i] -= 0.01 * g[i];
        }
    }

    println!("{:?}", db.w.to_vec());
}
