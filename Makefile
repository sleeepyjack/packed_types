BINDIR := bin
INCDIR := include

CC := g++
STD := c++14
NVCC := nvcc
CCFLAGS := -O3 -fopenmp
NVCCGENCODE = -gencode arch=compute_60,code=sm_60 \
			  -gencode arch=compute_61,code=sm_61 \
              -gencode arch=compute_70,code=sm_70
NVCCFLAGS := -std=$(STD) $(NVCCGENCODE) --expt-extended-lambda -ccbin $(CC) $(addprefix -Xcompiler ,$(CCFLAGS))

all: example

example: example.cu $(INCDIR)/packed_types.cuh $(INCDIR)/cudahelpers/cuda_helpers.cuh | $(BINDIR)
	$(NVCC) $(NVCCFLAGS) -I$(INCDIR) example.cu -o $(BINDIR)/example

debug: NVCCFLAGS += -g -O0 -Xptxas -v -UNDEBUG
debug: all

profile: NVCCFLAGS += -lineinfo -g -Xptxas -v
profile: all

clean:
	$(RM) -r $(BINDIR)

$(BINDIR):
	mkdir -p $@

.PHONY: clean all $$(BINDIR)

