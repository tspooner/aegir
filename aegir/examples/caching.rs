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

pub struct Ctx {
    pub x: Vec<f64>,
    cache: Option<f64>
}

impl Ctx {
    pub fn new(x: Vec<f64>) -> Ctx {
        Ctx { x, cache: None, }
    }
}

impl AsMut<Self> for Ctx {
    fn as_mut(&mut self) -> &mut Self { self }
}

impl Context for Ctx {}

impl Read<X> for Ctx {
    type Buffer = Vec<f64>;

    fn read(&self, _: X) -> Option<Vec<f64>> { Some(self.x.clone()) }
}

impl Read<C> for Ctx {
    type Buffer = f64;

    fn read(&self, _: C) -> Option<f64> { self.cache.clone() }
}

impl Write<C> for Ctx {
    fn write(&mut self, _: C, value: f64) { self.cache.replace(value); }
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
