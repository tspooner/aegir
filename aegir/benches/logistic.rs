#[macro_use]
extern crate aegir;
#[macro_use]
extern crate criterion;
extern crate rand;

use aegir::{
    buffers::{
        shapes::{S0, S1},
        Spec,
        Arrays,
        Buffer,
        Scalars,
    },
    ids::{W, X, Y},
    ops,
    Differentiable,
    Function,
    Identifier,
    Node,
    Read,
};
use criterion::{
    black_box,
    criterion_group,
    criterion_main,
    AxisScale,
    BenchmarkId,
    Criterion,
    PlotConfiguration,
};
use rand::{rngs::SmallRng, Rng, SeedableRng};

const TW: [f64; 20] = [
    0.5711859870, 0.4909409924, 0.5853200098, 0.1720337856, 0.4465294488, 0.7777692126,
    0.4422984780, 0.2736022852, 0.9573818740, 0.2796223361, 0.7067172602, 0.4888024657,
    0.6255499162, 0.8583838028, 0.6421767298, 0.2412597239, 0.6214162003, 0.1552535872,
    0.1265687185, 0.6177711088
];

ctx!(Ctx { x: X, y: Y, w: W });

macro_rules! solve {
    ([$n:literal] |$ctx:ident, $rng:ident| $grad:block) => {{
        let mut $rng = SmallRng::seed_from_u64(1994);
        let mut $ctx = Ctx {
            x: [0.0; $n],
            y: 0.0,
            w: [0.0; $n],
        };

        for _ in 0..1_000_000 {
            let g: [f64; $n] = $grad;

            for i in 0..$n {
                $ctx.w[i] += 0.001 * g[i];
            }
        }

        $ctx
    }};
}

macro_rules! solve_auto {
    ($n:literal) => {{
        let x = X.into_var();
        let y = Y.into_var();
        let w = W.into_var();

        let model = x.dot(w).sigmoid();
        let likelihood = model.mul(y).add(ops::OneSub(model).mul(ops::OneSub(y))).ln();
        let adj = likelihood.adjoint(W);

        solve!(
            [$n] |ctx, rng| {
                ctx.x = rng.gen::<[f64; $n]>();

                let yy = ctx.x.iter().zip(TW.iter().take($n)).map(|(x, tw)| x * tw).sum();

                ctx.y = rng.gen_bool(ops::sigmoid(yy)) as u8 as f64;

                adj.evaluate(&ctx).unwrap()
            }
        );
    }}
}

macro_rules! solve_manual {
    ($n:literal) => {{
        let x = X.into_var();
        let y = Y.into_var();
        let w = W.into_var();

        let model = x.dot(w).sigmoid();
        let l = y.mul(ops::OneSub(model));
        let r = ops::OneSub(y).mul(model);
        let adj = l.sub(r).mul(x);

        solve!(
            [$n] |ctx, rng| {
                ctx.x = rng.gen::<[f64; $n]>();

                let yy = ctx.x.iter().zip(TW.iter().take($n)).map(|(x, tw)| x * tw).sum();

                ctx.y = rng.gen_bool(ops::sigmoid(yy)) as u8 as f64;

                adj.evaluate(&ctx).unwrap()
            }
        );
    }}
}

pub fn benchmark_solve(c: &mut Criterion) {
    let mut group = c.benchmark_group("Logistic");

    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

    macro_rules! benches {
        ($n:literal) => {
            group.bench_with_input(BenchmarkId::new("Auto", $n), &$n, |b, _| {
                b.iter(|| { solve_auto!($n) })
            });
            group.bench_with_input(BenchmarkId::new("Manual", $n), &$n, |b, _| {
                b.iter(|| { solve_manual!($n) })
            });
        }
    }

    benches!(2);
    benches!(5);

    group.finish();
}

criterion_group!(logistic, benchmark_solve);
criterion_main!(logistic);
