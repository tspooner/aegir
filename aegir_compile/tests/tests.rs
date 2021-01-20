#[macro_use]
extern crate aegir;
extern crate aegir_compile;

use aegir::{sources::Constant, Identifier, Node};

ids!(A::a, B::b, C::c);

#[test]
fn test_constant_conversion() {
    let x1 = Constant::<i64>(10);
    let x2 = compile!(10);
    assert_eq!(x1, x2);

    let x3 = compile!(10.0);
    let x4 = Constant::<f64>(10.0);
    assert_eq!(x3, x4);
}

// TODO
// #[test]
// fn test_unary_operators() {
//     let x1 = Constant::<i64>(-10);
//     let x2 = compile!(-10);
//     assert_eq!(x1, x2);

//     let x3 = compile!(-10.0);
//     let x4 = Constant::<f64>(-10.0);
//     assert_eq!(x3, x4);
// }

#[test]
fn test_binary_operators() {
    let a = A.to_var();
    let b = B.to_var();

    let x1 = compile!(a + b);
    let x2 = a.add(b);
    assert_eq!(x1, x2);

    let x3 = compile!(a - b);
    let x4 = a.sub(b);
    assert_eq!(x3, x4);

    let x5 = compile!(a * b);
    let x6 = a.mul(b);
    assert_eq!(x5, x6);

    // TODO
    // let x7 = compile!(a / b);
    // let x8 = a.div(b);
    // assert_eq!(x7, x8);

    let x9 = compile!(a ^ b);
    let x10 = a.pow(b);
    assert_eq!(x9, x10);
}

#[test]
fn test_operator_chaining() {
    let a = A.to_var();
    let b = B.to_var();
    let c = C.to_var();

    let x1 = compile!(a + b * c);
    let x2 = a.add(b.mul(c));
    assert_eq!(x1, x2);

    let x3 = compile!(a * b + c);
    let x4 = a.mul(b).add(c);
    assert_eq!(x3, x4);
}
