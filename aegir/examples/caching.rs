#[macro_use]
extern crate aegir;
extern crate rand;

use aegir::{
    ids::{X, C},
    ops,
    Context,
    Differentiable,
    Function,
    Identifier,
    Read,
    Write,
    Node,
};
use std::time;

#[derive(Context)]
pub struct Ctx {
    #[id(X)] pub x: Vec<f64>,
    #[id(C)] #[cache] cache: Option<f64>,
}

impl Ctx {
    pub fn new(x: Vec<f64>) -> Ctx {
        Ctx { x, cache: None, }
    }
}

macro_rules! time_op {
    ($node:ident) => {{
        let repeated = $node
            .add($node.clone())
            .add($node.clone())
            .add($node.clone())
            .add($node.clone())
            .add($node.clone())
            .add($node.clone())
            .add($node.clone());

        let mut ctx = Ctx::new(vec![1.0; 100_000_000]);

        let start = time::Instant::now();

        repeated.evaluate(&mut ctx).ok();

        time::Instant::now().duration_since(start)
    }}
}

fn main() {
    let x = X.into_var();

    let basic = ops::Sum(ops::Negate(ops::Double(x)));
    let cached = basic.clone().cached(C);

    println!("Basic:\t{:?}", time_op!(basic));
    println!("Cached:\t{:?}", time_op!(cached));
}
