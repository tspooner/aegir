//! Module containing formatting logic.
use super::Node;

/// Helper type for propagating bracketing information.
///
/// Some expressions do not need to be surrounded by brackets. All of
/// this depends entirely on context, and this type helps propagate that
/// information for better formatting.
pub struct PreWrap {
    /// The internal contents of the string representation.
    pub text: String,

    /// This value is true if `text` needs surrounding brackets.
    pub needs_wrap: bool,
}

impl PreWrap {
    /// Return a wrapped string iff `needs_wrap` is true.
    pub fn to_safe_string(&self, l: char, r: char) -> String {
        if self.needs_wrap {
            format!("{}{}{}", l, self.text, r)
        } else {
            self.text.clone()
        }
    }
}

impl std::fmt::Display for PreWrap {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.text)
    }
}

/// Wrapper for expressions that can be written as strings.
pub enum Expr {
    /// An expression with known value zero.
    Zero,

    /// An expression with known value one.
    One,

    /// An expression with pre-computed string representation.
    Text(PreWrap),
}

impl std::fmt::Display for Expr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Expr::Zero => Ok(()),
            Expr::One => f.write_str("1"),
            Expr::Text(pw) => f.write_str(&pw.text),
        }
    }
}

/// Trait for operator nodes that can be expressed as a string.
pub trait ToExpr: Node {
    /// Convert the node to an expression string.
    ///
    /// # Examples
    ///
    /// ```
    /// # use aegir::{Identifier, fmt::ToExpr, ops::{Add, Mul}, ids::{X, Y, Z}};
    /// let op = Mul(Add(X.into_var(), Y.into_var()), Z.into_var());
    ///
    /// assert_eq!(op.to_expr().to_string(), "(x + y) ∘ z");
    /// ```
    fn to_expr(&self) -> Expr;
}
