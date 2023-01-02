# aegir

[![Crates.io](https://img.shields.io/crates/v/aegir.svg)](https://crates.io/crates/aegir)
[![Build Status](https://github.com/tspooner/aegir/actions/workflows/rust.yml/badge.svg)](https://github.com/tspooner/aegir/actions/workflows/rust.yml)

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
aegir = "2.0"
```

## Example

```rust
#[macro_use]
extern crate aegir;
extern crate rand;

use aegir::{Differentiable, Function, Identifier, Node, ids::{X, Y, W}};

ctx_type!(Ctx { x: X, y: Y, w: W });

fn main() {
    let mut rng = rand::thread_rng();
    let mut ctx = Ctx {
        x: [0.0; 2],
        y: 0.0,
        w: [0.0; 2],
    };

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

    for _ in 0..1_000_00 {
        // Independent variables:
        ctx.x = rng.gen();

        // Dependent variable:
        ctx.y = ctx.x[0] * 2.0 - ctx.x[1] * 4.0;

        // Evaluate gradient:
        let g: [f64; 2] = adj.evaluate(&ctx).unwrap();

        // Update weights:
        ctx.w[0] -= 0.01 * g[0];
        ctx.w[1] -= 0.01 * g[1];
    }

    println!("{:?}", ctx.w.to_vec());
}
```
