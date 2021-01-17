use proc_macro2::TokenStream;
use quote::ToTokens;

pub fn expand(ast: &syn::DeriveInput) -> TokenStream {
    match &ast.data {
        syn::Data::Struct(ref ds) => db_struct_impl(ast, ds),
        _ => unimplemented!(),
    }.into_token_stream()
}

fn parse_id_attr(attr_meta: &syn::Attribute) -> Result<syn::Ident, String> {
    match attr_meta.parse_meta() {
        Ok(syn::Meta::List(ml)) if ml.nested.len() > 1 =>
            Err("Too many arguments to #[id(.)]".to_string()),

        Ok(syn::Meta::List(ml)) if ml.nested.len() == 0 =>
            Err("Too few arguments to #[id(.)]".to_string()),

        Ok(syn::Meta::List(ml)) => match ml.nested.first().unwrap() {
            syn::NestedMeta::Meta(id_meta) => match id_meta {
                syn::Meta::Path(id_path) => Ok(id_path.get_ident().unwrap().to_owned()),

                _ => Err("Incorrectly formed argument to #[id(.)]".to_string()),
            },

            _ => Err("Literals not supported by #[id(.)]".to_string()),
        },

        _ => Err("Unsupported attribute format.".to_string())
    }
}

fn db_struct_impl(ast: &syn::DeriveInput, ds: &syn::DataStruct) -> TokenStream {
    let name = &ast.ident;
    let (impl_generics, ty_generics, where_clause) = ast.generics.split_for_impl();

    let mut code = quote! {
        impl #impl_generics ::aegir::Database for #name #ty_generics #where_clause {}
    };

    for f in &ds.fields {
        let ty = &f.ty;
        let field_name = &f.ident;

        let mut id_attrs = f.attrs
            .iter()
            .filter(|a| match a.path.get_ident() {
                Some(name) => name.to_string() == "id",
                None => false,
            });

        let field_id = parse_id_attr(id_attrs.next().unwrap()).unwrap();

        (quote! {
            impl #impl_generics ::aegir::Get<#field_id> for #name #ty_generics #where_clause {
                type Output = #ty;

                fn get(&self, _: #field_id) -> Option<&#ty> {
                    Some(&self.#field_name)
                }
            }
        }).to_tokens(&mut code);
    }

    code
}
