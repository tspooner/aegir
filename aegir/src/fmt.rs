use super::Node;

pub struct PreWrap {
    pub text: String,
    pub needs_wrap: bool,
}

impl PreWrap {
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

pub enum Expr {
    Zero,
    One,
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

pub trait ToExpr: Node {
    fn to_expr(&self) -> Expr;
}
