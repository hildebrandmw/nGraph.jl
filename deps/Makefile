mkfile_path := $(abspath $(lastword $(MAKEFILE_LIST)))
mkfile_dir := $(dir $(mkfile_path))

OPT= -g -O3
CXXFLAGS+= -Wall -fPIC $(OPT) -std=c++17 -DPCM_SILENT -DJULIA_ENABLE_THREADING -Dexcept_EXPORTS
CXXFLAGS+= -I$(CXXWRAP_HOME)/include -I$(JULIA_HOME)/include/julia -I$(mkfile_dir)usr/include

#CXXFLAGS+= -I$(mkfile_dir)/ngraph/build/mkldnn/include
# Add this flag to turn off a bunch of unused variable warnings in in libcxxwrap
CXXFLAGS+= -Wno-unused-variable -Wno-unused-lambda-capture
CXXFLAGS+= $(DEFINES)

#####
##### Hack to get lib directories better on Fedora
#####

libdir.x86_64 = lib64
libdir.i686   = lib

MACHINE := $(shell uname -m)

libdir = $(libdir.$(MACHINE))

# Linked Libraries:
#
# Julia
#
# - libjulia: Need access to Julia internals to get compilation with CxxWrap to work
# - libcxxwrap_julia: Access to C++ stuff from julia
#
# nGraph
#
# - libngraph: Kind of obvious, we need to link with libngraph to actually be able to do
#  		anything

LOAD_FLAGS+= -L$(JULIA_HOME)/lib/ -Wl,--export-dynamic -Wl,-rpath,$(JULIA_HOME)/lib -ljulia
LOAD_FLAGS+= -L$(CXXWRAP_HOME)/lib/ -Wl,-rpath,$(CXXWRAP_HOME)/lib -lcxxwrap_julia
LOAD_FLAGS+= -L$(mkfile_dir)usr/$(libdir) -Wl,-rpath,$(mkfile_dir)usr/$(libdir) -lngraph

.PHONY: all
all: libngraph-julia.so

libngraph-julia.so: ngraph-julia.cpp $(mkfile_dir)usr/$(libdir)/libngraph.so
	$(CXX) $(CXXFLAGS) -shared $^ -lpthread -o $@ $(LOAD_FLAGS)
