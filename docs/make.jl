using T4AITensorCompat
using Documenter

DocMeta.setdocmeta!(
    T4AITensorCompat,
    :DocTestSetup,
    :(using T4AITensorCompat);
    recursive = true,
)

makedocs(;
    modules = [T4AITensorCompat],
    authors = "H. Shinaoka <h.shinaoka@gmail.com>",
    sitename = "T4AITensorCompat.jl",
    format = Documenter.HTML(;
        canonical = "https://github.com/tensor4all/T4AITensorCompat.jl",
        edit_link = "main",
        assets = String[],
    ),
    pages = ["Home" => "index.md"],
    checkdocs = :exports,
)

deploydocs(; repo = "github.com/tensor4all/T4AITensorCompat.jl.git", devbranch = "main")
