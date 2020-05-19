# Note that this script can accept some limited command-line arguments, run
# `julia build_tarballs.jl --help` to see a usage message.
using BinaryBuilder, Pkg

name = "nGraph_jll"
version = v"0.1.0"

# Collection of sources required to complete build
sources = [
    GitSource("https://github.com/NervanaSystems/ngraph.git", "b8419c354e5fc70805f1501d7dfff533ac790bec")
]

# Bash recipe for building across all platforms
script = raw"""
cd $WORKSPACE/srcdir
cd ngraph
mkdir build
cd build
which clang
# Must set target to "" - otherwise, TBB gets confused
target="" cmake .. -DCMAKE_INSTALL_PREFIX=$prefix -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=/opt/bin/clang -DCMAKE_CXX_COMPILER=/opt/bin/clang++ 
make -j$(nproc)
make -j$(nproc)
make install
exit
"""

# These are the platforms we will build for by default, unless further
# platforms are passed in on the command line
platforms = [
    Linux(:x86_64, libc=:glibc)
]


# The products that we will ensure are always built
products = [
    LibraryProduct("libiomp5", :libiopm5),
    LibraryProduct("libdnnl", :libdnnl),
    LibraryProduct("libngraph", :libngraph),
    LibraryProduct("libnop_backend", :libnop_backend),
    LibraryProduct("libcodegen", :libcodegen),
    LibraryProduct("libcpu_backend", :libcpu_backend),
    LibraryProduct("libtbb", :libtbb),
    LibraryProduct("libgcpu_backend", :libgcpu_backend),
    LibraryProduct("libmklml_intel", :libmklml_intel),
    LibraryProduct("libinterpreter_backend", :libinterpreter_backend)
]

# Dependencies that must be installed before this package can be built
dependencies = Dependency[
]

# Build the tarballs, and possibly a `build.jl` as well.
build_tarballs(ARGS, name, version, sources, script, platforms, products, dependencies; preferred_gcc_version = v"6.1.0")
