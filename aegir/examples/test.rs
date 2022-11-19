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
    let sse = model.sub(y).squared();
    let adj = sse.adjoint(W);

    println!(
        "{:?}",
        adj.evaluate(Database {
            x: [0.0, 1.0],
            y: 5.0,
            w: &weights,
        })
        .unwrap()
    );
}
