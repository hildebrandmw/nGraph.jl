using CxxWrap, LibGit2, JSON

# We use the "build.json" file to control various build parameters for nGraph.
#
# Expected contents are:
#
# "PMDK" -> Bool: Build nGraph with PMDK support
# "DEBUG" -> Bool: Build DEBUG version of nGraph

#####
##### ngraph
#####

# Fetch repo
url = "https://github.com/darchr/ngraph"
branch = "master"

localdir = joinpath(@__DIR__, "ngraph")
ispath(localdir) || LibGit2.clone(url, localdir; branch = branch)

# Get build parameters
parameters = JSON.parsefile(joinpath(@__DIR__, "build.json"))

# build repo
builddir = joinpath(localdir, "build")
mkpath(builddir)
current_dir = pwd()

# nGraph is just generally happier if we build it with clang.
CC = "clang"
CXX = "clang++"

cd(builddir)
cmake_args = [
    "-DNGRAPH_ONNX_IMPORT_ENABLE=TRUE",
    "-DNGRAPH_CODEGEN_ENABLE=TRUE",
    "-DCMAKE_C_COMPILER=$CC",
    "-DCMAKE_CXX_COMPILER=$CXX",
    "-DCMAKE_INSTALL_PREFIX=$(joinpath(@__DIR__, "usr"))",
]

# Add additional parameters
parameters["PMDK"] && push!(cmake_args, "-DNGRAPH_PMDK_ENABLE=TRUE")
parameters["DEBUG"] && push!(cmake_args, "-DNGRAPH_DEBUG_ENABLE=TRUE")

run(`cmake .. $cmake_args`)
run(`make -j all`)
run(`make install`)

cd(current_dir)

#####
##### cxxwrap library
#####

println("Building Lib")

# Path to the CxxWrap dependencies
cxxhome = dirname(dirname(CxxWrap.jlcxx_path))
juliahome = dirname(Base.Sys.BINDIR)
run(`make JULIA_HOME=$juliahome CXXWRAP_HOME=$cxxhome CC=$CC CXX=$CXX -j all `)
