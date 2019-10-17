BINDIR := bin
INCDIR := include

CC := g++
STD := c++14
NVCC := nvcc
CCFLAGS := -O3
NVCCGENCODE = -arch=sm_35
NVCCFLAGS := -O3 -std=$(STD) $(NVCCGENCODE) -ccbin $(CC) $(addprefix -Xcompiler ,$(CCFLAGS))

all: example

example: example.cu $(INCDIR)/packed_types.cuh $(INCDIR)/cudahelpers/cuda_helpers.cuh | $(BINDIR)
	$(NVCC) $(NVCCFLAGS) -I$(INCDIR) example.cu -o $(BINDIR)/example

debug: NVCCFLAGS += -g -O0 -Xptxas -v -UNDEBUG -D_DEBUG
debug: all

profile: NVCCFLAGS += -lineinfo -g -Xptxas -v -DNDEBUG
profile: all

clean:
	$(RM) -r $(BINDIR)

$(BINDIR):
	mkdir -p $@

.PHONY: clean all $$(BINDIR)

