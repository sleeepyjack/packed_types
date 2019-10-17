SRCDIR := src
BINDIR := bin
INCDIR := ../include
TESTINCDIR := include
BUILDDIR := build

CC := g++
STD := c++14
NVCC := nvcc
CCFLAGS := -O3 -fopenmp
NVCCGENCODE = -arch=sm_35
NVCCFLAGS := -O3 -std=$(STD) $(NVCCGENCODE) --expt-extended-lambda -ccbin $(CC) $(addprefix -Xcompiler ,$(CCFLAGS))

SRC := $(wildcard $(SRCDIR)/*.cu)
OBJ := $(SRC:$(SRCDIR)/%.cu=$(BUILDDIR)/%.o)

all: test

test: $(OBJ) | $(BINDIR)
	$(NVCC) $(NVCCFLAGS) $(OBJ) -o $(BINDIR)/$@

$(BUILDDIR)/packed_types_test.o: $(SRCDIR)/packed_types_test.cu | $(BUILDDIR)
	$(NVCC) -I$(INCDIR) -I$(TESTINCDIR) $(NVCCFLAGS) -c $< -o $@

$(BUILDDIR)/catch_main.o: $(SRCDIR)/catch_main.cu | $(BUILDDIR)
	$(NVCC) -I$(TESTINCDIR) $(NVCCFLAGS) -c $< -o $@

debug: NVCCFLAGS += -g -O0 -Xptxas -v -UNDEBUG -D_DEBUG
debug: all

profile: NVCCFLAGS += -lineinfo -g -Xptxas -v -DUNDEBUG
profile: all

clean:
	$(RM) -r $(BUILDDIR) $(BINDIR)

$(BUILDDIR) $(BINDIR):
	mkdir -p $@

.PHONY: clean all $(BUILDDIR) $(BINDIR)

