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
branch = "mh/pmem"

# In order to use CUDA10, we need a version of Cmake > 3.12.
#
# The version that is installed on ubuntu by default is 3.10
#
# TODO: Automatically fetch cmake
cmake_path = joinpath(@__DIR__, "cmake", "bin", "cmake")

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
]

# Add additional parameters
parameters["PMDK"] && push!(cmake_args, "-DNGRAPH_PMDK_ENABLE=TRUE")
parameters["DEBUG"] && push!(cmake_args, "-DNGRAPH_DEBUG_ENABLE=TRUE")
parameters["GPU"] && push!(cmake_args, "-DNGRAPH_GPU_ENABLE=TRUE")


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

parameters["PMDK"] && push!(make_args, "DEFINES=-DNGRAPH_PMDK_ENABLE=TRUE")
run(`make $make_args -j $nproc `)
