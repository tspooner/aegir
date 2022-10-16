#[macro_use]
extern crate aegir;

use aegir::{Node, Identifier, Differentiable, ids::X};

fn main() {
    let x = X.into_var();
    let f = x.adjoint(X);
    let j = f.adjoint(X);

    println!("{}", f.is_zero());
    println!("{}", j.is_zero());
}
