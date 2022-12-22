#[macro_use]
extern crate aegir;
extern crate rand;

use aegir::{
    buffers::ZipMap,
    ids::{W, X, Y},
    Differentiable,
    Function,
    Identifier,
    Node,
};
use rand::{rngs::SmallRng, Rng, SeedableRng};

db!(Database { x: X, y: Y, w: W });

fn main() {
    let mut rng = SmallRng::seed_from_u64(1994);
    let mut db = Database {
        x: [0.0; 20],
        y: 0.0,
        w: [0.0; 20],
    };

    let true_weights: [f64; 20] = rng.gen();

    let x = X.into_var();
    let y = Y.into_var();
    let w = W.into_var();

    let model = x.dot(w);
    let sse = model.sub(y).squared();
    let adj = sse.adjoint(W);

    for _ in 0..1_000_000 {
        // Independent variables:
        db.x = rng.gen();

        // Dependent variable:
        db.y = db.x.iter().zip(true_weights.iter()).map(|(x, y)| x * y).sum();

        // Evaluate gradient:
        let g: [f64; 20] = adj.evaluate(&db).unwrap();

        for i in 0..20 {
            db.w[i] -= 0.01 * g[i];
        }
    }

    println!("{:?}", db.w.to_vec());
}
