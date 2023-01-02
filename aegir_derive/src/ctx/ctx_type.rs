use super::CtxField;
use proc_macro2::TokenStream;
use quote::ToTokens;
use syn::{braced, parse::{Parse, Parser, ParseStream}};

enum ParseError {
    InvalidModifier(syn::Ident),
}

impl std::fmt::Debug for ParseError {
    fn fmt<'a>(&self, f: &mut std::fmt::Formatter<'a>) -> std::fmt::Result {
        match self {
            ParseError::InvalidModifier(modi) => write!(f, "Invalid modifier: {}.", modi),
        }
    }
}

struct CtxFieldRaw {
    modifier: Option<syn::Ident>,
    name: syn::Ident,

    ident: syn::Ident,
}

impl CtxFieldRaw {
    fn normalise(self) -> Result<CtxField, ParseError> {
        let is_cache = self.modifier.map(|modi| if modi == "cache" {
            Ok(true)
        } else {
            Err(ParseError::InvalidModifier(modi))
        }).unwrap_or(Ok(false))?;

        let ident = self.ident;
        let buffer_ty = syn::Type::Path(syn::TypePath {
            qself: None,
            path: syn::PathSegment {
                ident: format_ident!("__{}", ident),
                arguments: syn::PathArguments::None,
            }.into(),
        });
        let ty = if is_cache { parse_quote! { Option<#buffer_ty> } } else { buffer_ty.clone() };

        Ok(CtxField {
            name: self.name,

            ty,
            buffer_ty,

            ident: Some(ident),
            is_cache,
        })
    }
}

impl Parse for CtxFieldRaw {
    fn parse(stream: ParseStream) -> syn::parse::Result<Self> {
        let name_or_modifier: syn::Ident = stream.parse()?;

        let name: syn::Ident;
        let modifier: Option<syn::Ident>;

        if stream.peek(syn::Ident) {
            name = stream.parse()?;
            modifier = Some(name_or_modifier);
        } else {
            name = name_or_modifier;
            modifier = None;
        }

        let _: syn::Token![:] = stream.parse()?;

        Ok(CtxFieldRaw {
            modifier,
            name,

            ident: stream.parse()?,
        })
    }
}

struct Struct {
    ident: syn::Ident,
    fields: syn::punctuated::Punctuated<CtxFieldRaw, Token![,]>,
}

impl Parse for Struct {
    fn parse(stream: ParseStream) -> syn::parse::Result<Self> {
        let ident: syn::Ident = stream.parse()?;

        let content;
        let _: syn::token::Brace = braced!(content in stream);

        let fields = content.parse_terminated(CtxFieldRaw::parse)?;

        Ok(Struct { ident, fields, })
    }
}

pub fn expand(tokens: proc_macro::TokenStream) -> proc_macro2::TokenStream {
    let s_def = Parser::parse(Struct::parse, tokens).unwrap();
    let name = s_def.ident;

    let fields: Vec<_> = s_def.fields.into_iter().map(|raw| raw.normalise().unwrap()) .collect();
    let fields_code = fields.iter().fold(TokenStream::new(), |mut code, field| {
        let id = field.ident.as_ref().unwrap();
        let attrs = if field.is_cache {
            quote! { #[cache] #[id(#id)] }
        } else {
            quote! { #[id(#id)] }
        };
        let name = &field.name;
        let ty = &field.ty;

        code.extend(quote! { #attrs #name: #ty, });

        code
    });

    let generics = syn::Generics {
        lt_token: Some(syn::token::Lt::default()),
        params: fields.iter().map(|field| {
            if let syn::Type::Path(tp) = &field.buffer_ty {
                syn::GenericParam::Type(tp.path.get_ident().unwrap().clone().into())
            } else {
                panic!("Invalid type encountered.");
            }
        }).collect(),
        gt_token: Some(syn::token::Gt::default()),
        where_clause: None,
    };
    let (_, ty_generics, _) = generics.split_for_impl();

    quote! {
        #[derive(Context)]
        pub struct #name #ty_generics {
            #fields_code
        }
    }.into_token_stream()
}
