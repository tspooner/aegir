#[derive(Debug)]
struct CtxField {
    pub name: syn::Ident,

    pub ty: syn::Type,
    pub buffer_ty: syn::Type,

    pub ident: Option<syn::Ident>,
    pub is_cache: bool,
}

pub mod derive;
pub mod ctx_type;
