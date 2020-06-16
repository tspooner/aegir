extern crate aegir;

use aegir::*;

fn main() {
    let x = aegir::Variable;
    let y = aegir::Neg(aegir::Constant(2.0).broadcast());

    // f(x, y) = cos(x - y)
    let f = aegir::Power(aegir::Cosine(aegir::Sum(x, y)), 2.0);

    let xval = [1.0, 3.05];
    let dual = f.dual((xval,));

    println!("{} = {:?}", f, dual.0);
    println!("∇({}) = {:?}", f, dual.1);
}
