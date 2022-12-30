use proc_macro2::TokenStream;
use quote::ToTokens;

pub fn expand(ast: &syn::DeriveInput) -> TokenStream {
    match &ast.data {
        syn::Data::Struct(ref ds) => ctx_struct_impl(ast, ds),
        _ => unimplemented!(),
    }
    .into_token_stream()
}

fn parse_id_attr(attr_meta: &syn::Attribute) -> Result<syn::Ident, String> {
    match attr_meta.parse_meta() {
        Ok(syn::Meta::List(ml)) if ml.nested.len() > 1 => {
            Err("Too many arguments to #[id(.)]".to_string())
        },

        Ok(syn::Meta::List(ml)) if ml.nested.len() == 0 => {
            Err("Too few arguments to #[id(.)]".to_string())
        },

        Ok(syn::Meta::List(ml)) => match ml.nested.first().unwrap() {
            syn::NestedMeta::Meta(id_meta) => match id_meta {
                syn::Meta::Path(id_path) => Ok(id_path.get_ident().unwrap().to_owned()),

                _ => Err("Incorrectly formed argument to #[id(.)]".to_string()),
            },

            _ => Err("Literals not supported by #[id(.)]".to_string()),
        },

        _ => Err("Unsupported attribute format.".to_string()),
    }
}

fn ctx_struct_impl(ast: &syn::DeriveInput, ds: &syn::DataStruct) -> TokenStream {
    let name = &ast.ident;
    let (impl_generics, ty_generics, where_clause) = ast.generics.split_for_impl();

    let mut code = quote! {
        impl #impl_generics ::std::convert::AsRef<Self> for #name #ty_generics #where_clause {
            fn as_ref(&self) -> &Self { self }
        }

        impl #impl_generics ::std::convert::AsMut<Self> for #name #ty_generics #where_clause {
            fn as_mut(&mut self) -> &mut Self { self }
        }

        impl #impl_generics ::aegir::Context for #name #ty_generics #where_clause {}
    };

    let id_fields = ds.fields.iter().filter_map(|f| {
        f.attrs
            .iter()
            .filter(|a| match a.path.get_ident() {
                Some(name) => name.to_string() == "id",
                None => false,
            })
            .next()
            .map(|attr| (f, attr))
    });

    for (f, attr) in id_fields {
        let ty = &f.ty;
        let field_id = parse_id_attr(attr).unwrap();
        let field_name = &f.ident;

        let mut c_generics = ast.generics.clone();
        let wc = c_generics
            .where_clause
            .get_or_insert_with(|| syn::WhereClause {
                where_token: syn::token::Where([proc_macro2::Span::call_site()]),
                predicates: syn::punctuated::Punctuated::new(),
            });

        wc.predicates.push(parse_quote! {
            #ty: Clone + ::aegir::buffers::IntoSpec + ::aegir::buffers::shapes::Shaped
        });
        wc.predicates.push(parse_quote! {
            <#ty as ::aegir::buffers::IntoSpec>::Buffer: ::aegir::buffers::shapes::Shaped<Shape = <#ty as ::aegir::buffers::shapes::Shaped>::Shape>
        });

        (quote! {
            impl #impl_generics ::aegir::Read<#field_id> for #name #ty_generics #wc {
                type Buffer = <#ty as ::aegir::buffers::IntoSpec>::Buffer;

                #[inline]
                fn read(&self, fid: #field_id) -> Option<Self::Buffer> {
                    self.read_spec(fid).map(|spec| spec.unwrap())
                }

                #[inline]
                fn read_spec(&self, _: #field_id) -> Option<::aegir::buffers::Spec<Self::Buffer>> {
                    use ::aegir::buffers::IntoSpec;

                    Some(self.#field_name.clone().into_spec())
                }

                #[inline]
                fn read_shape(&self, _: #field_id) -> Option<::aegir::buffers::shapes::ShapeOf<#ty>> {
                    use ::aegir::buffers::shapes::Shaped;

                    Some(self.#field_name.shape())
                }
            }
        })
        .to_tokens(&mut code);
    }

    code
}
