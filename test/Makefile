SRCDIR := src
BINDIR := bin
INCDIR := include
WCINCDIR := ../include
BUILDDIR := build

CC := g++
STD := c++14
NVCC := nvcc
CCFLAGS := -O3 -fopenmp
NVCCGENCODE = -gencode arch=compute_60,code=sm_60 \
			  -gencode arch=compute_61,code=sm_61 \
              -gencode arch=compute_70,code=sm_70
NVCCFLAGS := -std=$(STD) $(NVCCGENCODE) --expt-extended-lambda -ccbin $(CC) $(addprefix -Xcompiler ,$(CCFLAGS))

SRC := $(wildcard $(SRCDIR)/*.cu)
OBJ := $(SRC:$(SRCDIR)/%.cu=$(BUILDDIR)/%.o)
INC := $(wildcard $(INCDIR)/*)

all: test

test: $(OBJ) | $(BINDIR)
	$(NVCC) $(NVCCFLAGS) $(OBJ) -o $(BINDIR)/$@

$(BUILDDIR)/packed_types_test.o: $(SRCDIR)/packed_types_test.cu | $(BUILDDIR)
	$(NVCC) -I$(INCDIR) -I$(WCINCDIR) $(NVCCFLAGS) -c $< -o $@

$(BUILDDIR)/catch_main.o: $(SRCDIR)/catch_main.cu | $(BUILDDIR)
	$(NVCC) -I$(INCDIR) $(NVCCFLAGS) -c $< -o $@

debug: NVCCFLAGS += -g -O0 -Xptxas -v -UNDEBUG
debug: all

profile: NVCCFLAGS += -lineinfo -g -Xptxas -v
profile: all

clean:
	$(RM) -r $(BUILDDIR) $(BINDIR)

$(BUILDDIR) $(BINDIR):
	mkdir -p $@

.PHONY: clean all $(BUILDDIR) $(BINDIR)

