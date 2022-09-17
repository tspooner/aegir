# aegir

[![Crates.io](https://img.shields.io/crates/v/aegir.svg)](https://crates.io/crates/aegir)
[![Build Status](https://travis-ci.org/tspooner/aegir.svg?branch=master)](https://travis-ci.org/tspooner/aegir)
[![Coverage Status](https://coveralls.io/repos/github/tspooner/aegir/badge.svg?branch=master)](https://coveralls.io/github/tspooner/aegir?branch=master)

## Overview

> Strongly-typed, compile-time autodifferentiation in Rust.

`aegir` is an experimental autodifferentiation framework designed to leverage
the powerful type-system in Rust and _avoid runtime as much as humanly
possible_. The approach taken resembles that of expression templates, as
commonly used in linear-algebra libraries written in C++.

### Key Features
- Built-in arithmetic, linear-algebraic, trigonometric and special operators.
- Infinitely differentiable: _Jacobian, Hessian, etc..._
- Custom DSL for operator expansion.
- Decoupled/generic tensor type.

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

use aegir::{Differentiable, Function, Identifier, Node, ids::{X, Y, W}};

db!(Database { x: X, y: Y, w: W });

fn main() {
    let mut weights = [0.0, 0.0];

    let x = X.into_var();
    let y = Y.into_var();
    let w = W.into_var();

    let model = x.dot(w);

    // Using standard method calls...
    let sse = model.sub(y).squared();
    let adj = sse.adjoint(W);

    // ...or using aegir! macro
    let sse = aegir!((model - y) ^ 2);
    let adj = sse.adjoint(W);

    for _ in 0..100000 {
        let [x1, x2] = [rand::random::<f64>(), rand::random::<f64>()];

        let g = adj.evaluate(Database {
            // Independent variables:
            x: [x1, x2],

            // Dependent variable:
            y: x1 * 2.0 - x2 * 4.0,

            // Model weights:
            w: &weights,
        }).unwrap();

        weights[0] -= 0.01 * g[0][0];
        weights[1] -= 0.01 * g[0][1];
    }

    println!("{:?}", weights.to_vec());
}
```
