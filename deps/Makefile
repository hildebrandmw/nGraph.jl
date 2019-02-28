mkfile_path := $(abspath $(lastword $(MAKEFILE_LIST)))
mkfile_dir := $(dir $(mkfile_path))

OPT= -g -O3
CXXFLAGS+= -Wall -fPIC $(OPT) -std=c++14 -DPCM_SILENT -DJULIA_ENABLE_THREADING -Dexcept_EXPORTS 
CXXFLAGS+= -I$(CXXWRAP_HOME)/include -I$(JULIA_HOME)/include/julia -I$(mkfile_dir)usr/include
# Add this flag to turn off a bunch of unused variable warnings in in libcxxwrap
CXXFLAGS+= -Wno-unused-variable -Wno-unused-lambda-capture

LOAD_FLAGS+= -L$(JULIA_HOME)/lib/ -Wl,--export-dynamic -Wl,-rpath,$(JULIA_HOME)/lib -ljulia 
LOAD_FLAGS+= -L$(CXXWRAP_HOME)/lib/ -Wl,-rpath,$(CXXWRAP_HOME)/lib -lcxxwrap_julia  
LOAD_FLAGS+= -L$(mkfile_dir)usr/lib -Wl,-rpath,$(mkfile_dir)usr/lib -lngraph

all: libngraph-julia.so

libngraph-julia.so: ngraph-julia.cpp
	$(CXX) $(CXXFLAGS) -shared $^ -lpthread -o $@ $(LOAD_FLAGS)
