#[macro_use]
extern crate aegir;
extern crate rand;

use aegir::{
    ids::{W, X, Y},
    Differentiable,
    Function,
    Identifier,
    Node,
    ops,
};
use rand::{rngs::SmallRng, Rng, SeedableRng};

const N: usize = 5;

ctx!(Ctx { x: X, y: Y, w: W });

fn main() {
    let mut rng = SmallRng::seed_from_u64(1994);
    let mut db = Ctx {
        x: [1.0; N],
        y: 0.0,
        w: [0.0; N],
    };

    let true_weights: [f64; N] = rng.gen();

    let x = X.into_var();
    let y = Y.into_var();
    let w = W.into_var();

    let model = x.dot(w).sigmoid();
    let likelihood = model.mul(y).add(ops::OneSub(model).mul(ops::OneSub(y))).ln();
    let adj = likelihood.adjoint(W);

    for _ in 0..1_000_000 {
        // Independent variables:
        db.x = rng.gen();

        // Dependent variable:
        let yy = db.x.iter().zip(true_weights.iter()).map(|(x, tw)| x * tw).sum();

        db.y = rng.gen_bool(ops::sigmoid(yy)) as u8 as f64;

        // Evaluate gradient:
        let g: [f64; N] = adj.evaluate(&db).unwrap();

        for i in 0..N {
            db.w[i] += 0.0005 * g[i];
        }
    }

    println!("Target: {:?}", true_weights.to_vec());
    println!("Solution: {:?}", db.w.to_vec());
}
