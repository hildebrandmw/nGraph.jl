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
    #"-DNGRAPH_USE_PREBUILT_LLVM=TRUE",
    "-DCMAKE_BUILD_TYPE=Release",
    "-DCMAKE_C_COMPILER=$CC",
    "-DCMAKE_CXX_COMPILER=$CXX",
    "-DCMAKE_INSTALL_PREFIX=$(joinpath(@__DIR__, "usr"))",
    #"-DNGRAPH_TBB_ENABLE=FALSE", # causes build failure
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

# Path to the CxxWrap dependencies
cxxhome = CxxWrap.prefix_path()
juliahome = dirname(Base.Sys.BINDIR)
make_args = [
    "JULIA_HOME=$juliahome",
    "CXXWRAP_HOME=$cxxhome",
    "CC=$CC",
    "CXX=$CXX",
]

run(`make $make_args -j $nproc `)
