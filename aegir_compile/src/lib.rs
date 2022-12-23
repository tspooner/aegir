use proc_macro::TokenStream;
use quote::quote;
use syn::fold::Fold;
use syn::punctuated::Punctuated;
use syn::spanned::Spanned;
use syn::{parse_macro_input, parse_quote, token, BinOp, Expr, ExprMethodCall, Ident, Lit};

struct ConvertToAegir {}

impl Fold for ConvertToAegir {
    fn fold_expr(&mut self, expr: Expr) -> Expr {
        match expr {
            // Convert binary operations into method calls
            // Example: x + y --> x.add(y)
            Expr::Binary(b) => {
                let span = b.span();

                let method = match b.op {
                    BinOp::Add(_) => "add",
                    BinOp::Sub(_) => "sub",
                    BinOp::Mul(_) => "mul",
                    // TODO:
                    BinOp::Div(_) => unimplemented!(),
                    BinOp::BitXor(_) => "pow",
                    _ => unimplemented!(),
                };

                fn is_square(lit: &Lit) -> bool {
                    match lit {
                        Lit::Float(f) => {
                            let two: syn::LitFloat = parse_quote!(2.0);

                            f == &two
                        },
                        Lit::Int(f) => {
                            let two: syn::LitInt = parse_quote!(2);

                            f == &two
                        },
                        _ => false,
                    }
                }

                match *b.right {
                    Expr::Lit(ref l) if method == "pow" && is_square(&l.lit) => {
                        Expr::MethodCall(ExprMethodCall {
                            attrs: b.attrs,
                            receiver: Box::new(self.fold_expr(*b.left)),
                            dot_token: token::Dot { spans: [span] },
                            method: Ident::new("squared", span),
                            turbofish: None,
                            paren_token: token::Paren { span },
                            args: Punctuated::<Expr, token::Comma>::new(),
                        })
                    },
                    _ => {
                        let mut args = Punctuated::<Expr, token::Comma>::new();

                        args.push_value(self.fold_expr(*b.right));

                        Expr::MethodCall(ExprMethodCall {
                            attrs: b.attrs,
                            receiver: Box::new(self.fold_expr(*b.left)),
                            dot_token: token::Dot { spans: [span] },
                            method: Ident::new(method, span),
                            turbofish: None,
                            paren_token: token::Paren { span },
                            args,
                        })
                    },
                }
            },
            // Convert integer and float literals into Constant type instances
            // Example: 10 --> Constant<i64>(10)
            Expr::Lit(l) => match l.lit {
                Lit::Int(val) => {
                    parse_quote!({
                        aegir::Constant::<i64>(#val)
                    })
                },
                Lit::Float(val) => {
                    parse_quote!({
                        aegir::Constant::<f64>(#val)
                    })
                },
                _ => Expr::Lit(l),
            },
            Expr::Array(_binding_0) => Expr::Array(self.fold_expr_array(_binding_0)),
            Expr::Assign(_binding_0) => Expr::Assign(self.fold_expr_assign(_binding_0)),
            Expr::AssignOp(_binding_0) => Expr::AssignOp(self.fold_expr_assign_op(_binding_0)),
            Expr::Async(_binding_0) => Expr::Async(self.fold_expr_async(_binding_0)),
            Expr::Await(_binding_0) => Expr::Await(self.fold_expr_await(_binding_0)),
            // Expr::Binary(_binding_0) => Expr::Binary(self.fold_expr_binary(_binding_0)),
            Expr::Block(_binding_0) => Expr::Block(self.fold_expr_block(_binding_0)),
            Expr::Box(_binding_0) => Expr::Box(self.fold_expr_box(_binding_0)),
            Expr::Break(_binding_0) => Expr::Break(self.fold_expr_break(_binding_0)),
            Expr::Call(_binding_0) => Expr::Call(self.fold_expr_call(_binding_0)),
            Expr::Cast(_binding_0) => Expr::Cast(self.fold_expr_cast(_binding_0)),
            Expr::Closure(_binding_0) => Expr::Closure(self.fold_expr_closure(_binding_0)),
            Expr::Continue(_binding_0) => Expr::Continue(self.fold_expr_continue(_binding_0)),
            Expr::Field(_binding_0) => Expr::Field(self.fold_expr_field(_binding_0)),
            Expr::ForLoop(_binding_0) => Expr::ForLoop(self.fold_expr_for_loop(_binding_0)),
            Expr::Group(_binding_0) => Expr::Group(self.fold_expr_group(_binding_0)),
            Expr::If(_binding_0) => Expr::If(self.fold_expr_if(_binding_0)),
            Expr::Index(_binding_0) => Expr::Index(self.fold_expr_index(_binding_0)),
            Expr::Let(_binding_0) => Expr::Let(self.fold_expr_let(_binding_0)),
            // Expr::Lit(_binding_0) => Expr::Lit(self.fold_expr_lit(_binding_0)),
            Expr::Loop(_binding_0) => Expr::Loop(self.fold_expr_loop(_binding_0)),
            Expr::Macro(_binding_0) => Expr::Macro(self.fold_expr_macro(_binding_0)),
            Expr::Match(_binding_0) => Expr::Match(self.fold_expr_match(_binding_0)),
            Expr::MethodCall(_binding_0) => {
                Expr::MethodCall(self.fold_expr_method_call(_binding_0))
            },
            Expr::Paren(_binding_0) => Expr::Paren(self.fold_expr_paren(_binding_0)),
            Expr::Path(_binding_0) => Expr::Path(self.fold_expr_path(_binding_0)),
            Expr::Range(_binding_0) => Expr::Range(self.fold_expr_range(_binding_0)),
            Expr::Reference(_binding_0) => Expr::Reference(self.fold_expr_reference(_binding_0)),
            Expr::Repeat(_binding_0) => Expr::Repeat(self.fold_expr_repeat(_binding_0)),
            Expr::Return(_binding_0) => Expr::Return(self.fold_expr_return(_binding_0)),
            Expr::Struct(_binding_0) => Expr::Struct(self.fold_expr_struct(_binding_0)),
            Expr::Try(_binding_0) => Expr::Try(self.fold_expr_try(_binding_0)),
            Expr::TryBlock(_binding_0) => Expr::TryBlock(self.fold_expr_try_block(_binding_0)),
            Expr::Tuple(_binding_0) => Expr::Tuple(self.fold_expr_tuple(_binding_0)),
            Expr::Type(_binding_0) => Expr::Type(self.fold_expr_type(_binding_0)),
            Expr::Unary(_binding_0) => Expr::Unary(self.fold_expr_unary(_binding_0)),
            Expr::Unsafe(_binding_0) => Expr::Unsafe(self.fold_expr_unsafe(_binding_0)),
            Expr::Verbatim(_binding_0) => Expr::Verbatim(_binding_0),
            Expr::While(_binding_0) => Expr::While(self.fold_expr_while(_binding_0)),
            Expr::Yield(_binding_0) => Expr::Yield(self.fold_expr_yield(_binding_0)),
            _ => unreachable!(),
        }
    }
}

#[proc_macro]
pub fn aegir(tokens: TokenStream) -> TokenStream {
    let expr: Expr = parse_macro_input!(tokens as Expr);

    let mut converter = ConvertToAegir {};
    let converted = converter.fold_expr(expr);

    TokenStream::from(quote!(#converted))
}
