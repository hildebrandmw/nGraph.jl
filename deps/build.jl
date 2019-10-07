using CxxWrap, LibGit2, JSON

# We use the "build.json" file to control various build parameters for nGraph.
#
# Expected contents are:
#
# "PMDK" -> Bool: Build nGraph with PMDK support
# "DEBUG" -> Bool: Build DEBUG version of nGraph

#####
##### cmake
#####

# Need to install a more recent version of cmake to support building with CUDA 10
cmake_path = joinpath(@__DIR__, "cmake", "bin", "cmake")
if !ispath(cmake_path)
    download(
        "https://github.com/Kitware/CMake/releases/download/v3.14.5/cmake-3.14.5-Linux-x86_64.tar.gz",
        "cmake.tar.gz",
    )
    run(`tar -xvf cmake.tar.gz`)
    mv("cmake-3.14.5-Linux-x86_64", "cmake", force = true)
end

#####
##### ngraph
#####

# Fetch repo
url = "https://github.com/darchr/ngraph"
#branch = "mh/autotm"
branch = "mh/pmem"

localdir = joinpath(@__DIR__, "ngraph")
ispath(localdir) || LibGit2.clone(url, localdir; branch = branch)

# Get build parameters
parameters = JSON.parsefile(joinpath(@__DIR__, "build.json"))

# build repo
#
# Separate out "debug" and "build" directories since "debug" tends to spam a lot of output
# and I don't want to have to recompile EVERYTHING each time I switch between the two.
if parameters["DEBUG"]
    builddir = joinpath(localdir, "debug")
else
    builddir = joinpath(localdir, "build")
end

mkpath(builddir)
current_dir = pwd()

# nGraph is just generally happier if we build it with clang.
CC = "clang"
CXX = "clang++"
nproc = parse(Int, read(`nproc`, String))

cd(builddir)
cmake_args = [
    "-DNGRAPH_CODEGEN_ENABLE=TRUE",
    "-DCMAKE_BUILD_TYPE=Release",
    #"-DNGRAPH_TBB_ENABLE=FALSE",
    "-DCMAKE_C_COMPILER=$CC",
    "-DCMAKE_CXX_COMPILER=$CXX",
    "-DCMAKE_INSTALL_PREFIX=$(joinpath(@__DIR__, "usr"))",
    "-DNGRAPH_USE_LEGACY_MKLDNN=FALSE",
    #"-DNGRAPH_TBB_ENABLE=FALSE",   # errors during build
]

# Add additional parameters
parameters["PMDK"] && push!(cmake_args, "-DNGRAPH_PMDK_ENABLE=TRUE")
parameters["DEBUG"] && push!(cmake_args, "-DNGRAPH_DEBUG_ENABLE=TRUE")
parameters["GPU"] && push!(cmake_args, "-DNGRAPH_GPU_ENABLE=TRUE")
parameters["NUMA"] && push!(cmake_args, "-DNGRAPH_NUMA_ENABLE=TRUE")


run(`$cmake_path .. $cmake_args`)
run(`make -j $nproc`)
run(`make install`)

cd(current_dir)

#####
##### cxxwrap library
#####

@info "Building Lib"

# Path to the CxxWrap dependencies
cxxhome = dirname(dirname(CxxWrap.jlcxx_path))
juliahome = dirname(Base.Sys.BINDIR)
make_args = [
    "JULIA_HOME=$juliahome",
    "CXXWRAP_HOME=$cxxhome",
    "CC=$CC",
    "CXX=$CXX",
]

# Define some macros to control switches in `ngraph-julia.cpp`
defines = String[]
parameters["PMDK"] && push!(defines, "-DNGRAPH_PMDK_ENABLE=TRUE")
parameters["GPU"] && push!(defines, "-DNGRAPH_GPU_ENABLE=TRUE")
if !isempty(defines)
    push!(make_args, "DEFINES=\"$(join(defines, " "))\"")
end

run(`make $make_args -j $nproc `)
