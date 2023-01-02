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

ctx_type!(Ctx {
    x: X,
    cache cc: C,
});

impl Ctx<Vec<f64>, f64> {
    pub fn new(x: Vec<f64>) -> Self {
        Ctx { x, cc: None, }
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
