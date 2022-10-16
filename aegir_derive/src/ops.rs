use proc_macro2::{Span, TokenStream};
use quote::ToTokens;
use std::convert::TryFrom;

#[derive(Debug)]
pub enum Error {
    ParseError,
}

#[derive(Debug)]
pub struct OpAttributes {}

impl TryFrom<&syn::Field> for OpAttributes {
    type Error = Error;

    fn try_from(f: &syn::Field) -> Result<OpAttributes, Error> {
        let op_attrs: Vec<_> = f
            .attrs
            .iter()
            .filter_map(|a| a.path.get_ident().filter(|id| id == &"op"))
            .collect();

        if op_attrs.len() != 1 {
            Err(Error::ParseError)
        } else {
            Ok(OpAttributes {})
        }
    }
}

#[derive(Debug)]
pub struct OpField {
    pub ty: syn::Type,
    pub accessor: proc_macro2::TokenStream,
    pub attributes: OpAttributes,
}

#[derive(Debug)]
pub struct OpFields(pub Vec<OpField>);

impl OpFields {
    pub fn extend_generics(
        &self,
        mut generics: syn::Generics,
        where_bounds: syn::punctuated::Punctuated<syn::TypeParamBound, Token![+]>,
    ) -> syn::Generics {
        let wc = generics
            .where_clause
            .get_or_insert_with(|| syn::WhereClause {
                where_token: syn::token::Where([Span::call_site()]),
                predicates: syn::punctuated::Punctuated::new(),
            });

        for of in self.iter() {
            let of_ty = &of.ty;

            wc.predicates.push(parse_quote! { #of_ty: #where_bounds });
        }

        generics
    }
}

impl std::iter::FromIterator<OpField> for OpFields {
    fn from_iter<I: IntoIterator<Item = OpField>>(iter: I) -> OpFields {
        OpFields(iter.into_iter().collect())
    }
}

impl std::ops::Deref for OpFields {
    type Target = Vec<OpField>;

    fn deref(&self) -> &Self::Target { &self.0 }
}

impl TryFrom<&syn::Data> for OpFields {
    type Error = Error;

    fn try_from(data: &syn::Data) -> Result<OpFields, Error> {
        fn parse_fields_named(fields: &syn::FieldsNamed) -> Result<OpFields, Error> {
            fields
                .named
                .iter()
                .map(|f| {
                    OpAttributes::try_from(f).map(|oas| OpField {
                        ty: f.ty.clone(),
                        accessor: f.ident.to_token_stream(),
                        attributes: oas,
                    })
                })
                .collect()
        }

        fn parse_fields_unnamed(fields: &syn::FieldsUnnamed) -> Result<OpFields, Error> {
            fields
                .unnamed
                .iter()
                .enumerate()
                .map(|(i, f)| {
                    OpAttributes::try_from(f).map(|oas| OpField {
                        ty: f.ty.clone(),
                        accessor: syn::Index {
                            index: i as u32,
                            span: Span::call_site(),
                        }
                        .into_token_stream(),
                        attributes: oas,
                    })
                })
                .collect()
        }

        match data {
            syn::Data::Struct(ds) => match &ds.fields {
                syn::Fields::Named(fields) => parse_fields_named(fields),
                syn::Fields::Unnamed(fields) => parse_fields_unnamed(fields),
                _ => todo!(),
            },
            _ => todo!(),
        }
    }
}

pub fn expand_contains(ast: &syn::DeriveInput) -> TokenStream {
    let name = &ast.ident;
    let op_fields = OpFields::try_from(&ast.data).expect("struct should have #[op] annotated successor nodes");

    let mut trait_generics = op_fields.extend_generics(
        ast.generics.clone(),
        parse_quote! { ::aegir::Contains<__TARGET> },
    );

    trait_generics
        .params
        .push(parse_quote! { __TARGET: ::aegir::Identifier });

    let (impl_generics, _, where_clause) = trait_generics.split_for_impl();
    let (_, ty_generics, _) = ast.generics.split_for_impl();

    let predicate: syn::punctuated::Punctuated<_, Token![||]> = op_fields
        .iter()
        .map(|of| {
            let ident = &of.accessor;

            quote! { self.#ident.contains(target) }
        })
        .collect();

    quote! {
        impl #impl_generics ::aegir::Contains<__TARGET> for #name #ty_generics #where_clause {
            fn contains(&self, target: __TARGET) -> bool { #predicate }
        }
    }
}
