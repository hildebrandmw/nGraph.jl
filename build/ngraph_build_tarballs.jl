# Note that this script can accept some limited command-line arguments, run
# `julia build_tarballs.jl --help` to see a usage message.
using BinaryBuilder, Pkg

name = "ngraph"
version = v"0.0.1"

# Collection of sources required to complete build
sources = [
    GitSource("https://github.com/NervanaSystems/ngraph.git", "81ca5be950bb62b97c5f6f96616c9de5b39ebf45"),
    GitSource("https://github.com/Kitware/Cmake.git", "dee2eff2cfb4d0743c5b2c3468e2b9227baff102")
]

# Bash recipe for building across all platforms
script = raw"""
cd $WORKSPACE/srcdir
cd Cmake
mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=$prefix -DCMAKE_TOOLCHAIN_FILE=${CMAKE_TARGET_TOOLCHAIN} -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
make install
cd ${WORKSPACE}/srcdir/ngraph
mkdir build
cd build
${WORKSPACE}/destdir/bin/cmake ..     -DCMAKE_INSTALL_PREFIX=$prefix     -DCMAKE_TOOLCHAIN_FILE=${CMAKE_TARGET_TOOLCHAIN}     -DCMAKE_BUILD_TYPE=Release     -DNGRAPH_TBB_ENABLE=false     -DNGRAPH_CPU_CODEGEN_ENABLE=true
export PATH="${PATH}:$(pwd)/src/resource"
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
    LibraryProduct("libdnnl", :libdnnl),
    LibraryProduct("libinterpreter_backend", :libinterpreter_backend),
    LibraryProduct("libnop_backend", :libnop_backend),
    LibraryProduct("libiomp5", :libiomp5),
    LibraryProduct("libmklml_intel", :libmklml_intel),
    LibraryProduct("libcodegen", :libcodegen),
    LibraryProduct("libeval_backend", :libeval_backend),
    LibraryProduct("libcpu_backend", :libcpu_backend),
    LibraryProduct("libngraph", :libngraph)
]

# Dependencies that must be installed before this package can be built
dependencies = [
    Dependency(PackageSpec(name="OpenSSL_jll", uuid="458c3c95-2e84-50aa-8efc-19380b2a3a95"))
]

# Build the tarballs, and possibly a `build.jl` as well.
build_tarballs(ARGS, name, version, sources, script, platforms, products, dependencies; preferred_gcc_version = v"8.1.0")
