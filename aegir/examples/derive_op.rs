#[macro_use]
extern crate aegir;

use aegir::Contains;

ids!(X::x, Y::y);

#[derive(Node, Contains)]
pub struct AddOne<N> {
    #[op] inner: N,
}

fn main() {
    let op = AddOne { inner: aegir::sources::Variable(X) };

    println!("{:?}", op.contains(X));
    println!("{:?}", op.contains(Y));
}
