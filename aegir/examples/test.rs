extern crate aegir;

fn main() {
    let dual = aegir::dual::Dual::variable(1.0);

    println!("{:?}", -dual - 1.0);
}
