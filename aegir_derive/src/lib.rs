extern crate proc_macro;
#[macro_use]
extern crate quote;
#[macro_use]
extern crate syn;

use proc_macro::TokenStream;

mod ids;

mod ctx;
mod ops;

#[proc_macro]
pub fn ids(tokens: TokenStream) -> TokenStream { ids::expand(tokens).into() }

#[proc_macro_derive(Context, attributes(id))]
pub fn derive_ctx(tokens: TokenStream) -> TokenStream {
    let ast = syn::parse2(tokens.into()).unwrap();

    ctx::expand(&ast).into()
}

#[proc_macro_derive(Contains, attributes(op))]
pub fn derive_contains(tokens: TokenStream) -> TokenStream {
    let ast = syn::parse2(tokens.into()).unwrap();

    ops::expand_contains(&ast).into()
}
