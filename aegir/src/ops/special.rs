use crate::{
    buffers::Buffer,
    Function,
};
use special_fun::FloatSpecial;

// Derive = x.gamma() * x.digamma()
impl_unary!(
    Gamma<F: FloatSpecial>, |x| { x.gamma() }, |self| {
        use crate::fmt::{PreWrap, Expr::*};

        match self.0.to_expr() {
            Zero => Text(PreWrap {
                text: "\u{221E}".to_string(),
                needs_wrap: false,
            }),
            One => One,
            Text(pw) => Text(PreWrap {
                text: format!("\u{0393}({})", pw),
                needs_wrap: false,
            })
        }
    }
);

// Deriv = x.digamma()
impl_unary!(
    LogGamma<F: FloatSpecial>, |x| { x.loggamma() }, |self| {
        use crate::fmt::{PreWrap, Expr::*};

        match self.0.to_expr() {
            Zero => Text(PreWrap {
                text: "\u{221E}".to_string(),
                needs_wrap: false,
            }),
            One => Zero,
            Text(pw) => Text(PreWrap {
                text: format!("ln\u{0393}({})", pw),
                needs_wrap: false,
            })
        }
    }
);

// Deriv = x.loggamma()
impl_unary!(
    Factorial<F: FloatSpecial>, |x| { x.factorial() }, |self| {
        use crate::fmt::{PreWrap, Expr::*};

        match self.0.to_expr() {
            Zero | One => One,
            Text(pw) => Text(PreWrap {
                text: format!("{}!", pw.to_safe_string('(', ')')),
                needs_wrap: false,
            })
        }
    }
);

impl_unary!(
    Erf<F: FloatSpecial>, |x| { x.erf() }, |self| {
        use crate::fmt::{PreWrap, Expr::*};

        match self.0.to_expr() {
            Zero | One => One,
            Text(pw) => Text(PreWrap {
                text: format!("erf({})", pw),
                needs_wrap: false,
            })
        }
    }
);

impl<N> Erf<N> {
    pub fn complementary(self) -> crate::ops::Negate<Self> { crate::ops::Negate(self) }
}
