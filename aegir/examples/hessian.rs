#[macro_use]
extern crate aegir;
extern crate rand;

use aegir::{
    ids::{W, X},
    Differentiable,
    Function,
    Identifier,
    Node,
};

db!(Database { x: X, w: W });

fn main() {
    let x = X.into_var();
    let w = W.into_var();

    let f = aegir::ops::Sin(x);
    let j_x = f.adjoint(X);
    let j_x_x = j_x.adjoint(X);

    let db = Database {
        x: 3.0f64,
        w: [-5.0f64, 6.0f64],
    };

    println!("f = {} = {:?}", f, f.evaluate(&db).unwrap());
    println!("j_x = {} = {:?}", j_x, j_x.evaluate(&db).unwrap());
    println!("j_x_x = {} = {:?}", j_x_x, j_x_x.evaluate(&db).unwrap());
}
