using CxxWrap, LibGit2

#####
##### Fetch Repo
#####

# Go through Master - then checkout the correct tab.
url = "https://github.com/NervanaSystems/ngraph"
branch = "master"
tag = "v0.29.0-rc.0"

localdir = joinpath(@__DIR__, "ngraph")
if !ispath(localdir)
    LibGit2.clone(url, localdir; branch = branch)
    repo = LibGit2.GitRepo(localdir)
    commit = LibGit2.GitCommit(repo, tag)
    LibGit2.checkout!(repo, string(LibGit2.GitHash(commit)))
end

#####
##### building
#####

builddir = joinpath(localdir, "build")
mkpath(builddir)
current_dir = pwd()

# nGraph is just generally happier if we build it with clang.
CC = "clang"
CXX = "clang++"
nproc = parse(Int, read(`nproc`, String))

cd(builddir)
cmake_args = [
    "-DCMAKE_BUILD_TYPE=Release",
    "-DCMAKE_C_COMPILER=$CC",
    "-DCMAKE_CXX_COMPILER=$CXX",
    "-DCMAKE_INSTALL_PREFIX=$(joinpath(@__DIR__, "usr"))",
]

run(`cmake .. $cmake_args`)
run(`make -j $nproc`)
run(`make install`)

cd(current_dir)

# On Fedora, we also need to symlink `libmkldnn` for things to work properly.
if ispath(joinpath(current_dir, "usr", "lib64"))
    run(`ln -sf "$current_dir/usr/lib64/libmkldnn.so" "$current_dir/usr/lib64/libmkldnn.so.0"`)
end

#####
##### cxxwrap library
#####
@info "Building Lib"

cxxhome = CxxWrap.prefix_path()
juliahome = dirname(Base.Sys.BINDIR)

# Use Clang since it seems to get along better with Julia
cxx = "clang++"

cxxflags = [
    "-g",
    "-O3",
    "-Wall",
    "-fPIC",
    "-std=c++17",
    "-DPCM_SILENT",
    "-DJULIA_ENABLE_THREADING",
    "-Dexcept_EXPORTS",
    # Surpress some warnings from Cxx
    "-Wno-unused-variable",
    "-Wno-unused-lambda-capture",
]

includes = [
    "-I$(joinpath(cxxhome, "include"))",
    "-I$(joinpath(juliahome, "include", "julia"))",
    "-I$(joinpath(@__DIR__, "usr", "include"))",
]

_libpath = joinpath(current_dir, "usr", "lib")
_lib64path = joinpath(current_dir, "usr", "lib64")

libpath = ispath(_libpath) ? _libpath : _lib64path

loadflags = [
    # Linking flags for Julia
    "-L$(joinpath(juliahome, "lib"))",
    "-Wl,--export-dynamic",
    "-Wl,-rpath,$(joinpath(juliahome, "lib"))",
    "-ljulia",
    # Linking Flags for CxxWrap
    "-L$(joinpath(cxxhome, "lib"))",
    "-Wl,-rpath,$(joinpath(cxxhome, "lib"))",
    "-lcxxwrap_julia",
    # Linking Flags for nGraph
    "-L$libpath",
    "-Wl,-rpath,$libpath",
    "-lngraph",
]


src = joinpath(current_dir, "ngraph-julia.cpp")
so = joinpath(current_dir, "libngraph-julia.so")

cmd = `$cxx $cxxflags $includes -shared $src -lpthread -o $so $loadflags`
@show cmd
run(cmd)


#run(`make $make_args -j $nproc `)
