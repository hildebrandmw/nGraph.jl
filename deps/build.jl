using CxxWrap
using ngraph_jll

@info "Building Lib"

# Paths for linking
cxxhome = CxxWrap.prefix_path()
juliahome = dirname(Base.Sys.BINDIR)
ngraphhome = dirname(dirname(ngraph_jll.libngraph_path))

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
    "-I$(joinpath(ngraphhome, "include"))",
]

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
    "-L$(joinpath(ngraphhome, "lib"))",
    "-Wl,-rpath,$(joinpath(ngraphhome, "lib"))",
    "-lngraph",
]

src = joinpath(@__DIR__, "ngraph-julia.cpp")
so = joinpath(@__DIR__, "libngraph-julia.so")

cmd = `$cxx $cxxflags $includes -shared $src -lpthread -o $so $loadflags`
@show cmd
run(cmd)

