extern crate rand;

fn main() {
    let mut weights = [0.0, 0.0];

    for _ in 0..1_000_000 {
        let [x1, x2] = [rand::random::<f64>(), rand::random::<f64>()];

        let y = x1 * 2.0 - x2 * 4.0;
        let p = weights[0] * x1 + weights[1] * x2;
        let e = p - y;
        let g = [
            2.0 * x1 * e,
            2.0 * x2 * e
        ];

        weights[0] -= 0.01 * g[0];
        weights[1] -= 0.01 * g[1];
    }

    println!("{:?}", weights.to_vec());
}
