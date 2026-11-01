ifneq ($(COSMOSIS_SRC_DIR),)
include ${COSMOSIS_SRC_DIR}/config/compilers.mk
endif

$(info $$MPIFC is ${MPIFC})
ifeq ($(strip $(MPIFC)),)
MPI =
else
MPI = $(shell command -v $(MPIFC) 2> /dev/null)
endif

TARGETS=libnest3.so
ifeq ($(MPI),)
    $(info MPI Compiler not found)
else
    $(info MPI Compiler found so compiling MPI multinest)
    TARGETS += libnest3_mpi.so

    # gfortran 10 does not cope with some behaviour of
    # MPICH (argument mismatches).
    # check if we are using such a version and if so add a legacy flag.
    # We could probably get away with adding that everywhere but I still
    # hold out vague hopes of supporting e.g. intel fortran sometime.
    IS_GFORTRAN=$(shell $(MPIFC) --version | grep -q "GNU Fortran"; echo $$?)
    ifeq ($(IS_GFORTRAN),0)
        $(info Using gfortran)
        GF_VERSION_GT_10=$(shell expr `$(MPIFC) -dumpversion | cut -f1 -d.` \>= 10)
        ifeq ($(GF_VERSION_GT_10),1)
            $(info "Using gfortran version >= 10 so adding -std=legacy")
            FFLAGS += -std=legacy
        endif

    endif

endif


ifeq ($(COSMOSIS_DEBUG),1)
FFLAGS += -fPIC -ffree-line-length-none
else
FFLAGS += -O3 -fPIC -ffree-line-length-none
endif

# Linux loaders reject shared objects that request an executable stack.
# Keep this flag platform-specific because Darwin's linker does not accept
# the GNU ELF option.
ifeq ($(shell uname -s),Linux)
STACK_FLAGS = -Wl,-z,noexecstack
else
STACK_FLAGS =
endif

ifeq ($(LAPACK_LINK),)
LAPACKLIB = -llapack
else
LAPACKLIB = $(LAPACK_LINK)
endif

NESTLIBDIR = ./

NSOBJECTS = utils.o utils1.o priors.o kmeans_clstr.o xmeans_clstr.o posterior.o nested.o cwrapper.o
NSOBJECTS_MPI = $(patsubst %.o,mpi/%.o,$(NSOBJECTS))

all: $(TARGETS)
	echo Built $(TARGETS)

%.o: %.f90
	$(FC) $(FFLAGS) -cpp -c -o $@ $^

mpi/%.o: %.f90 mpi/.phony
	$(MPIFC) $(FFLAGS) -cpp -DMPI -c -o $@ $<

mpi/.phony:
	@mkdir -p mpi
	@touch mpi/.phony

test:
	@echo "Alas, MultiNest has no tests"

libnest3_mpi.so: $(NSOBJECTS_MPI)
	echo LAPACKLIB is $(LAPACKLIB)
	$(MPIFC) -shared -o  $(CURDIR)/$@ $^ $(LIBS) $(LAPACKLIB) $(FFLAGS) $(LDFLAGS) $(STACK_FLAGS)
	
libnest3.so: $(NSOBJECTS) 
	echo LAPACKLIB is $(LAPACKLIB)
	$(FC) -shared -o  $(CURDIR)/$@ $^ $(LIBS) $(LAPACKLIB) $(FFLAGS)  $(LDFLAGS) $(STACK_FLAGS)

obj_detect:
	make -C example_obj_detect

gaussian:
	make -C example_gaussian

rosenbrock:
	make -C example_rosenbrock

ackley:
	make -C example_ackley

himmelblau:
	make -C example_himmelblau

eggbox:
	make -C example_eggbox

gauss_shell:
	make -C example_gauss_shell

eggboxC:
	make -C example_eggbox_C

eggboxC++:
	make -C example_eggbox_C++

clean: 
	-rm -f $(NESTLIBDIR)/libnest3_mpi.*
	-rm -rf mpi
	-rm -f $(NESTLIBDIR)/libnest3.* *.o *.mod

cleanall: clean_exec clean clean_obj_detect clean_gaussian clean_gauss_shell clean_eggbox \
clean_example_eggbox_C clean_example_eggbox_C++ clean_rosenbrock clean_himmelblau \
clean_ackley

clean_exec:
	-rm obj_detect gaussian rosenbrock ackley himmelblau gauss_shell eggbox eggboxC eggboxC++

clean_obj_detect:
	make -C example_obj_detect clean

clean_gaussian:
	make -C example_gaussian clean

clean_rosenbrock:
	make -C example_rosenbrock clean

clean_ackley:
	make -C example_ackley clean

clean_himmelblau:
	make -C example_himmelblau clean

clean_eggbox:
	make -C example_eggbox clean

clean_gauss_shell:
	make -C example_gauss_shell clean

clean_example_eggbox_C:
	make -C example_eggbox_C clean

clean_example_eggbox_C++:
	make -C example_eggbox_C++ clean
