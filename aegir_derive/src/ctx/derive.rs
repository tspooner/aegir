use super::CtxField;
use proc_macro2::TokenStream;
use quote::ToTokens;
use syn::{self, visit::Visit};

// Entry point:
pub fn expand(di: syn::DeriveInput) -> TokenStream {
    match di.data {
        syn::Data::Struct(ds) => ctx_struct_impl(di.ident, di.generics, ds),
        _ => unimplemented!(),
    }
    .into_token_stream()
}

// Implementation:
enum IdParseError {
    MetaParseError(syn::parse::Error),

    TooFewArguments,
    TooManyArguments,

    Improper,
    Literal,
    Unsupported,
}

impl std::fmt::Debug for IdParseError {
    fn fmt<'a>(&self, f: &mut std::fmt::Formatter<'a>) -> std::fmt::Result {
        use IdParseError::*;

        match self {
            MetaParseError(err) => err.fmt(f),

            TooFewArguments => write!(f, "Too few arguments to #[id(.)]"),
            TooManyArguments => write!(f, "Too many arguments to #[id(.)]"),

            Improper => write!(f, "Incorrectly formed argument to #[id(.)]"),
            Literal => write!(f, "Literals not supported by #[id(.)]"),
            Unsupported => write!(f, "Unsupported attribute format."),
        }
    }
}

enum AttrParseError {
    InvalidAttr,

    IdAttr(IdParseError),
}

impl std::fmt::Debug for AttrParseError {
    fn fmt<'a>(&self, f: &mut std::fmt::Formatter<'a>) -> std::fmt::Result {
        use AttrParseError::*;

        match self {
            InvalidAttr => write!(f, "Invalid attribute."),

            IdAttr(err) => err.fmt(f),
        }
    }
}

fn parse_id_attr(attr: &syn::Attribute) -> Result<syn::Ident, IdParseError> {
    use IdParseError::*;

    match attr.parse_meta().map_err(MetaParseError)? {
        syn::Meta::List(ml) if ml.nested.len() == 0 => { Err(TooFewArguments) },
        syn::Meta::List(ml) if ml.nested.len() > 1 => { Err(TooManyArguments) },

        syn::Meta::List(ml) => match ml.nested.first() {
            Some(syn::NestedMeta::Meta(id_meta)) => match id_meta {
                syn::Meta::Path(id_path) => Ok(id_path.get_ident().unwrap().to_owned()),

                _ => Err(Improper),
            },

            _ => Err(Literal),
        },

        _ => Err(Unsupported),
    }
}

enum CtxAttr {
    IdAttr(syn::Ident),
    CacheAttr,

    Unknown,
}

impl CtxAttr {
    fn parse_attr(attr: &syn::Attribute) -> Result<Self, AttrParseError> {
        if let Some(ident) = attr.path.get_ident() {
            if ident == "cache" {
                Ok(CtxAttr::CacheAttr)
            } else if ident == "id" {
                parse_id_attr(attr).map(CtxAttr::IdAttr).map_err(AttrParseError::IdAttr)
            } else {
                Ok(CtxAttr::Unknown)
            }
        } else {
            Err(AttrParseError::InvalidAttr)
        }
    }
}

enum CtxFieldError {
    ParseError(AttrParseError),

    NoFieldId,
    TooManyIds,

    ImproperCacheType,
}

impl std::fmt::Debug for CtxFieldError {
    fn fmt<'a>(&self, f: &mut std::fmt::Formatter<'a>) -> std::fmt::Result {
        use CtxFieldError::*;

        match self {
            ParseError(err) => err.fmt(f),

            NoFieldId => write!(f, "Field is labelled as #[cache] but has no #[id(.)] attr."),
            TooManyIds => write!(f, "Too many #[id(.)] attributes provided."),

            ImproperCacheType => write!(f, "Cache field must be of type Option<Buffer>."),
        }
    }
}

fn parse_ty_cache(ty: &syn::Type) -> Result<syn::Type, CtxFieldError> {
    if let syn::Type::Path(tp) = ty {
        tp.path.segments
            .first()
            .filter(|outer_ty| outer_ty.ident == "Option")
            .and_then(|outer_ty| {
                use syn::PathArguments::AngleBracketed;

                if let AngleBracketed(ab) = &outer_ty.arguments {
                    let ab_args = &ab.args;

                    Some(syn::Type::Path(syn::TypePath {
                        qself: tp.qself.clone(),
                        path: syn::Path {
                            leading_colon: tp.path.leading_colon.clone(),
                            segments: parse_quote! { #ab_args },
                        },
                    }))
                } else {
                    None
                }
            })
            .ok_or(CtxFieldError::ImproperCacheType)
    } else {
        Err(CtxFieldError::ImproperCacheType)
    }
}

fn parse_field(field: &syn::Field) -> Result<CtxField, CtxFieldError> {
    let name = field.ident.to_owned().ok_or(CtxFieldError::NoFieldId)?;

    let mut is_cache = false;
    let mut field_ident = None;

    let valid_attrs = field.attrs
        .iter()
        .map(|attr| CtxAttr::parse_attr(attr))
        .collect::<Result<Vec<CtxAttr>, _>>()
        .map_err(CtxFieldError::ParseError)?;

    for ctx_attr in valid_attrs {
        match ctx_attr {
            CtxAttr::Unknown => {},
            CtxAttr::CacheAttr => { is_cache = true; }
            CtxAttr::IdAttr(ident) => if field_ident.is_some() {
                return Err(CtxFieldError::TooManyIds);
            } else {
                field_ident = Some(ident);
            },
        }
    }

    if is_cache && field_ident.is_none() {
        Err(CtxFieldError::NoFieldId)
    } else {
        let buffer_ty = if is_cache {
            parse_ty_cache(&field.ty)?
        } else {
            field.ty.to_owned()
        };

        Ok(CtxField {
            name,

            ty: field.ty.to_owned(),
            buffer_ty,

            ident: field_ident,
            is_cache,
        })
    }
}

fn emit_read_fns(ctx_field: &CtxField) -> Option<TokenStream> {
    let field_name = &ctx_field.name;
    let field_bty = &ctx_field.buffer_ty;

    ctx_field.ident.as_ref().map(|field_id| if ctx_field.is_cache {
        quote! {
            #[inline]
            fn read_spec(&self, _: #field_id) -> Option<::aegir::buffers::Spec<#field_bty>> {
                use ::aegir::buffers::IntoSpec;

                self.#field_name.clone().map(|buf| buf.into_spec())
            }

            #[inline]
            fn read_shape(&self, _: #field_id) -> Option<::aegir::buffers::shapes::ShapeOf<#field_bty>> {
                use ::aegir::buffers::shapes::Shaped;

                self.#field_name.as_ref().map(|buf| buf.shape())
            }
        }
    } else {
        quote! {
            #[inline]
            fn read_spec(&self, _: #field_id) -> Option<::aegir::buffers::Spec<#field_bty>> {
                use ::aegir::buffers::IntoSpec;

                Some(self.#field_name.clone().into_spec())
            }

            #[inline]
            fn read_shape(&self, _: #field_id) -> Option<::aegir::buffers::shapes::ShapeOf<#field_bty>> {
                use ::aegir::buffers::shapes::Shaped;

                Some(self.#field_name.shape())
            }
        }
    })
}

fn emit_write_fns(ctx_field: &CtxField) -> Option<TokenStream> {
    let field_name = &ctx_field.name;

    ctx_field.ident.as_ref().map(|field_id| if ctx_field.is_cache {
        quote! {
            #[inline]
            fn write(&mut self, fid: #field_id, value: Self::Buffer) {
                self.#field_name.replace(value);
            }
        }
    } else {
        quote! {
            #[inline]
            fn write(&mut self, fid: #field_id, value: Self::Buffer) {
                self.#field_name = value;
            }
        }
    })
}

struct FieldExtractor(Vec<CtxField>);

impl FieldExtractor {
    fn new() -> Self { Self(vec![]) }

    fn fields(ds: &syn::DataStruct) -> Vec<CtxField> {
        let mut mutator = Self::new();

        mutator.visit_data_struct(ds);

        mutator.0
    }
}

impl<'a> Visit<'a> for FieldExtractor {
    fn visit_field(&mut self, field: &syn::Field) {
        self.0.push(parse_field(&field).unwrap());
    }
}

fn ctx_struct_impl(name: syn::Ident, generics: syn::Generics, ds: syn::DataStruct) -> TokenStream {
    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();

    let mut code = quote! {
        impl #impl_generics ::std::convert::AsRef<Self> for #name #ty_generics #where_clause {
            fn as_ref(&self) -> &Self { self }
        }

        impl #impl_generics ::std::convert::AsMut<Self> for #name #ty_generics #where_clause {
            fn as_mut(&mut self) -> &mut Self { self }
        }

        impl #impl_generics ::aegir::Context for #name #ty_generics #where_clause {}
    };

    for field in FieldExtractor::fields(&ds) {
        let field_id = match &field.ident {
            Some(id) => id,
            _ => continue,
        };
        let buffer_ty = &field.buffer_ty;

        let mut wc = generics.where_clause.clone();
        let wc_ref = wc.get_or_insert_with(|| syn::WhereClause {
            where_token: syn::token::Where([proc_macro2::Span::call_site()]),
            predicates: syn::punctuated::Punctuated::new(),
        });

        wc_ref.predicates.push(parse_quote! { #buffer_ty: Clone + ::aegir::buffers::Buffer });

        let read_fns = emit_read_fns(&field).unwrap();
        let write_fns = emit_write_fns(&field).unwrap();

        let impls = quote! {
            impl #impl_generics ::aegir::Read<#field_id> for #name #ty_generics #wc {
                type Buffer = <#buffer_ty as ::aegir::buffers::IntoSpec>::Buffer;

                #[inline]
                fn read(&self, fid: #field_id) -> Option<Self::Buffer> {
                    self.read_spec(fid).map(|spec| spec.unwrap())
                }

                #read_fns
            }

            impl #impl_generics ::aegir::Write<#field_id> for #name #ty_generics #wc {
                #write_fns
            }
        };

        impls.to_tokens(&mut code);
    }

    code.into_token_stream()
}
