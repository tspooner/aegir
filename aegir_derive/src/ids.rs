use proc_macro2::TokenStream;
use quote::ToTokens;
use syn::parse::Parser;

struct DataIdentifier {
    pub type_id: syn::Ident,
    pub colon: Option<Token![::]>,
    pub symbol: Option<syn::Ident>,
}

impl DataIdentifier {
    pub fn to_struct(&self) -> syn::ItemStruct {
        let type_id = &self.type_id;

        syn::parse2(quote! {
            #[derive(Clone, Copy, Debug, PartialEq, Eq)]
            pub struct #type_id;
        })
        .unwrap()
    }

    pub fn impl_identifier(&self) -> syn::ItemImpl {
        let type_id = &self.type_id;

        syn::parse2(quote! { impl ::aegir::Identifier for #type_id {} }).unwrap()
    }

    pub fn impl_display(&self) -> syn::ItemImpl {
        let type_id = &self.type_id;
        let symbol = self.symbol.as_ref().unwrap_or(type_id);

        syn::parse2(quote! {
            impl std::fmt::Display for #type_id {
                fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                    write!(f, stringify!(#symbol))
                }
            }
        })
        .unwrap()
    }

    pub fn impl_partial_eq(&self, other: &DataIdentifier, code: TokenStream) -> syn::ItemImpl {
        let type_id = &self.type_id;
        let type_id_ = &other.type_id;

        syn::parse2(quote! {
            impl std::cmp::PartialEq<#type_id_> for #type_id {
                fn eq(&self, other: &#type_id_) -> bool { #code }
            }
        })
        .unwrap()
    }
}

impl syn::parse::Parse for DataIdentifier {
    fn parse(input: syn::parse::ParseStream) -> syn::parse::Result<DataIdentifier> {
        let type_id: syn::Ident = input.parse()?;
        let lookahead = input.lookahead1();

        if lookahead.peek(Token![::]) {
            Ok(DataIdentifier {
                type_id,
                colon: Some(input.parse()?),
                symbol: Some(input.parse()?),
            })
        } else {
            Ok(DataIdentifier {
                type_id,
                colon: None,
                symbol: None,
            })
        }
    }
}

type DataIdentifiers = syn::punctuated::Punctuated<DataIdentifier, Token![,]>;

pub fn expand(tokens: proc_macro::TokenStream) -> proc_macro2::TokenStream {
    let data_ids = DataIdentifiers::parse_separated_nonempty
        .parse(tokens)
        .unwrap();

    let mut code = TokenStream::new();

    for data_id in data_ids.iter() {
        data_id.to_struct().to_tokens(&mut code);
        data_id.impl_identifier().to_tokens(&mut code);
        data_id.impl_display().to_tokens(&mut code);

        for data_id_ in data_ids.iter().filter(|di| di.type_id != data_id.type_id) {
            data_id
                .impl_partial_eq(data_id_, quote! { false })
                .to_tokens(&mut code);
        }
    }

    code
}
