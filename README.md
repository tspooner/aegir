# aegir

[![Crates.io](https://img.shields.io/crates/v/aegir.svg)](https://crates.io/crates/aegir)
[![Build Status](https://travis-ci.org/tspooner/aegir.svg?branch=master)](https://travis-ci.org/tspooner/aegir)
[![Coverage Status](https://coveralls.io/repos/github/tspooner/aegir/badge.svg?branch=master)](https://coveralls.io/github/tspooner/aegir?branch=master)

## Overview

`aegir` is a strongly-typed, reverse-mode autodiff library in Rust which
facilitates end-to-end, DAG-based computation.

## Installation

```toml
[dependencies]
aegir = "0.1"
```

## Example

```rust
#[macro_use]
extern crate aegir;
extern crate rand;

use aegir::{Node, Identifier, Differentiable};
use ndarray::Array1;

ids!(X::x, Y::y, W::w);
state!(State {
    input: X,
    output: Y,
    weights: W
});

macro_rules! get_state {
    ($x:ident, $y:ident, $w:ident) => {
        &State { input: &$x, output: &$y, weights: &$w, }
    }
}

fn main() {
    let mut weights = Array1::from(vec![0.0; 1]);

    let x = X.to_var();
    let y = Y.to_var();
    let w = W.to_var();

    let sse = y.sub(w.dot(x)).pow(2.0).reduce();

    for _ in 0..10000 {
        let x_ = Array1::from(vec![rand::random::<f64>()]);
        let y_ = x_.clone() * 2.0;
        let g = sse.grad(W, get_state!(x_, y_, weights)).unwrap();

        weights.iter_mut().zip(g.iter()).for_each(|(w, dw)| { *w -= 0.01 * dw });
    }

    println!("{:?}", weights.to_vec());
}
```
