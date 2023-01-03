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

ctx_type!(Ctx { x: X, y: Y, w: W });

macro_rules! solve {
    ([$n:literal] |$ctx:ident, $xs:ident| $grad:block) => {{
        let mut rng = SmallRng::seed_from_u64(1994);
        let mut $ctx = Ctx {
            x: [0.0; $n],
            y: 0.0,
            w: [0.0; $n],
        };

        for $xs in (0..1_000_000).map(|_| rng.gen::<[f64; $n]>()) {
            let g: [f64; $n] = $grad;

            for i in 0..$n {
                $ctx.w[i] -= 0.01 * g[i];
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

        let model = x.dot(w);
        let sse = model.sub(y).squared();
        let adj = sse.adjoint(W);

        solve!(
            [$n] |ctx, xs| {
                ctx.y = xs.iter().zip(TW.iter().take($n)).fold(0.0, |acc, (x, y)| acc + x * y);
                ctx.x = xs;

                adj.evaluate(&mut ctx).unwrap()
            }
        );
    }}
}

macro_rules! solve_manual {
    ($n:literal) => {{
        let x = X.into_var();
        let y = Y.into_var();
        let w = W.into_var();

        let adj = ops::Mul(ops::Double(ops::Sub(ops::TensorDot::new(x, w), y)), x);

        solve!(
            [$n] |ctx, xs| {
                ctx.y = xs.iter().zip(TW.iter().take($n)).fold(0.0, |acc, (x, y)| acc + x * y);
                ctx.x = xs;

                adj.evaluate(&mut ctx).unwrap()
            }
        );
    }}
}

pub struct RawAdjoint<const N: usize>;

impl<const N: usize> Node for RawAdjoint<N> {}

impl<const N: usize, C> Function<C> for RawAdjoint<N>
where
    C: Read<X, Buffer = [f64; N]> + Read<W, Buffer = [f64; N]> + Read<Y, Buffer = f64>,
{
    type Error = aegir::errors::NoError;
    type Value = [f64; N];

    fn evaluate<CR: AsMut<C>>(&self, mut ctx: CR) -> Result<Self::Value, Self::Error> {
        let ctxr = ctx.as_mut();

        let xs = ctxr.read(X).unwrap();
        let ws = ctxr.read(W).unwrap();
        let y = ctxr.read(Y).unwrap();

        let mut p = 0.0;

        for i in 0..N {
            p += xs[i] * ws[i];
        }

        let e = 2.0 * (p - y);

        Ok(xs.map(|x| x * e))
    }
}

macro_rules! solve_raw {
    ($n:literal) => {{
        let adj: RawAdjoint<$n> = RawAdjoint;

        solve!(
            [$n] |ctx, xs| {
                ctx.y = xs.iter().zip(TW.iter().take($n)).fold(0.0, |acc, (x, y)| acc + x * y);
                ctx.x = xs;

                adj.evaluate(&mut ctx).unwrap()
            }
        );
    }}
}

pub fn benchmark_solve(c: &mut Criterion) {
    let mut group = c.benchmark_group("Linear");

    group.plot_config(PlotConfiguration::default().summary_scale(AxisScale::Logarithmic));

    macro_rules! benches {
        ($n:literal) => {
            group.bench_with_input(BenchmarkId::new("Auto", $n), &$n, |b, _| {
                b.iter(|| { solve_auto!($n) })
            });
            group.bench_with_input(BenchmarkId::new("Manual", $n), &$n, |b, _| {
                b.iter(|| { solve_manual!($n) })
            });
            group.bench_with_input(BenchmarkId::new("Raw", $n), &$n, |b, _| {
                b.iter(|| { solve_raw!($n) })
            });
        }
    }

    benches!(2);
    benches!(5);
    benches!(10);
    benches!(20);

    group.finish();
}

criterion_group!(linear, benchmark_solve);
criterion_main!(linear);
