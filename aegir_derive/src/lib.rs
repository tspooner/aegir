extern crate proc_macro;
#[macro_use]
extern crate quote;
#[macro_use]
extern crate syn;

use proc_macro::TokenStream;

mod state;
mod ids;

#[proc_macro_derive(State, attributes(id))]
pub fn derive_state(tokens: TokenStream) -> TokenStream {
    let ast = syn::parse2(tokens.into()).unwrap();

    state::expand(&ast).into()
}

#[proc_macro]
pub fn ids(tokens: TokenStream) -> TokenStream {
    ids::expand(tokens).into()
}
