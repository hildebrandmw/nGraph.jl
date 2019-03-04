using CxxWrap, LibGit2

#####
##### ngraph
#####

# Fetch repo
url = "https://github.com/darchr/ngraph"
#branch = "master"
branch = "mh/persistent-malloc"

localdir = joinpath(@__DIR__, "ngraph")
ispath(localdir) || LibGit2.clone(url, localdir; branch = branch)

# build repo
builddir = joinpath(localdir, "build")
mkpath(builddir)
current_dir = pwd()

CC = "clang"
CXX = "clang++"

cd(builddir)
cmake_args = [
    "-DNGRAPH_ONNX_IMPORT_ENABLE=TRUE",
    "-DNGRAPH_CODEGEN_ENABLE=TRUE",
    "-DNGRAPH_PMDK_ENABLE=TRUE",
    #"-DNGRAPH_DEBUG_ENABLE=TRUE",
    #"-DNGRAPH_TARGET_ARCH=skylake-avx512",
    # I was getting segfaults during code generation. Trying to build with clang to see
    # if that works better (the code generator used by ngraph is llvm based, so they'll 
    # probably ... hopefully ... be a little more compatible)
    "-DCMAKE_C_COMPILER=$CC",
    "-DCMAKE_CXX_COMPILER=$CXX",
    "-DCMAKE_INSTALL_PREFIX=$(joinpath(@__DIR__, "usr"))",
]

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
