#[macro_use]
extern crate aegir;
#[macro_use]
extern crate criterion;
extern crate rand;

use aegir::{
    ids::{W, X, Y},
    buffers::{Buffer, Scalars, Arrays, shapes::{S0, S1}},
    ops,
    Read,
    Differentiable,
    Function,
    Identifier,
    Node,
};
use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, PlotConfiguration, AxisScale};
use rand::{Rng, SeedableRng, rngs::SmallRng};

db!(Database { x: X, y: Y, w: W });

pub fn benchmark_eval(c: &mut Criterion) {
    let mut group = c.benchmark_group("evaluate");

    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

    group.bench_function(BenchmarkId::new("LR", "NonZero"), |b| {
        let x = X.into_var();
        let y = Y.into_var();
        let w = W.into_var();

        let model = x.dot(w);
        let sse = model.sub(y).squared();
        let adj = sse.adjoint(W);

        b.iter(move || {
            adj.evaluate(black_box(Database {
                x: [1.0, 2.0],
                y: 3.0,
                w: [1.0, 1.0]
            }))
        })
    });
    group.bench_function(BenchmarkId::new("LR", "ZeroedWeights"), |b| {
        let x = X.into_var();
        let y = Y.into_var();
        let w = W.into_var();

        let model = x.dot(w);
        let sse = model.sub(y).squared();
        let adj = sse.adjoint(W);

        b.iter(move || {
            adj.evaluate(black_box(Database {
                x: [1.0, 2.0],
                y: 3.0,
                w: [0.0, 0.0]
            }))
        })
    });
    group.bench_function(BenchmarkId::new("LR", "ZeroedInput"), |b| {
        let x = X.into_var();
        let y = Y.into_var();
        let w = W.into_var();

        let model = x.dot(w);
        let sse = model.sub(y).squared();
        let adj = sse.adjoint(W);

        b.iter(move || {
            adj.evaluate(black_box(Database {
                x: [0.0, 0.0],
                y: 0.0,
                w: [1.0, 1.0]
            }))
        })
    });
    group.bench_function(BenchmarkId::new("LR", "ZeroedAll"), |b| {
        let x = X.into_var();
        let y = Y.into_var();
        let w = W.into_var();

        let model = x.dot(w);
        let sse = model.sub(y).squared();
        let adj = sse.adjoint(W);

        b.iter(move || {
            adj.evaluate(black_box(Database {
                x: [0.0, 0.0],
                y: 0.0,
                w: [0.0, 0.0]
            }))
        })
    });

    group.finish();
}

macro_rules! solve {
    ([$n:expr] |$db:ident, $xs:ident| $grad:block) => {{
        let mut rng = SmallRng::seed_from_u64(1994);
        let mut $db = Database {
            x: [0.0, 0.0],
            y: 0.0,
            w: [0.0, 0.0],
        };

        for $xs in (0..$n).map(|_| rng.gen::<[f64; 2]>()) {
            let g = $grad;

            $db.w[0] -= 0.01 * g[0];
            $db.w[1] -= 0.01 * g[1];
        }

        $db
    }};
}

fn solve_auto(n: usize) {
    let x = X.into_var();
    let y = Y.into_var();
    let w = W.into_var();

    let model = x.dot(w);
    let sse = model.sub(y).squared();
    let adj = sse.adjoint(W);

    solve!([n] |db, xs| {
        db.y = xs[0] * 2.0 - xs[1] * 4.0;
        db.x = xs;

        adj.evaluate(&db).unwrap()
    });
}

fn solve_manual(n: usize) {
    let x = X.into_var();
    let y = Y.into_var();
    let w = W.into_var();

    let adj = ops::Mul(ops::Double(ops::Sub(ops::TensorDot::new(x, w), y)), x);

    solve!([n] |db, xs| {
        db.y = xs[0] * 2.0 - xs[1] * 4.0;
        db.x = xs;

        adj.evaluate(&db).unwrap()
    });
}

pub struct RawAdjoint;

impl Node for RawAdjoint {}

impl<D> Function<D> for RawAdjoint
where
    D: Read<X, Buffer = [f64; 2]> + Read<W, Buffer = [f64; 2]> + Read<Y, Buffer = f64>,
{
    type Error = aegir::NoError;
    type Value = [f64; 2];

    fn evaluate<DR: AsRef<D>>(&self, db: DR) -> Result<Self::Value, Self::Error> {
        let dbr = db.as_ref();

        let xs = dbr.read(X).unwrap();
        let ws = dbr.read(W).unwrap();
        let y = dbr.read(Y).unwrap();

        let y = xs[0] * 2.0 - xs[1] * 4.0;
        let p = ws[0] * xs[0] + ws[1] * xs[1];
        let e = p - y;

        Ok([2.0 * xs[0] * e, 2.0 * xs[1] * e])
    }
}

fn solve_raw(n: usize) {
    let adj = RawAdjoint;

    solve!([n] |db, xs| {
        db.y = xs[0] * 2.0 - xs[1] * 4.0;
        db.x = xs;

        adj.evaluate(&db).unwrap()
    });
}

pub fn benchmark_solve(c: &mut Criterion) {
    let mut group = c.benchmark_group("solve");

    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

    for n in [10_000, 1_000_000] {
        group.bench_with_input(BenchmarkId::new("Auto", n), &n, |b, &n| {
            b.iter(|| solve_auto(n))
        });
        group.bench_with_input(BenchmarkId::new("Manual", n), &n, |b, &n| {
            b.iter(|| solve_manual(n))
        });
        group.bench_with_input(BenchmarkId::new("Raw", n), &n, |b, &n| {
            b.iter(|| solve_raw(n))
        });
    }

    group.finish();
}

criterion_group!(lr_sgd, benchmark_solve, benchmark_eval);
criterion_main!(lr_sgd);
