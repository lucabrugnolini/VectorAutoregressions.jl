using Documenter
using VectorAutoregressions

makedocs(
    sitename = "VectorAutoregressions",
    format = Documenter.HTML(),
    modules = [VectorAutoregressions]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#=deploydocs(
    repo = "<repository url>"
)=#
