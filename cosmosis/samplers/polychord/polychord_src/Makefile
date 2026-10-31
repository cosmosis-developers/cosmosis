ifneq ($(COSMOSIS_SRC_DIR),)
include ${COSMOSIS_SRC_DIR}/config/compilers.mk
endif

MPI = $(shell which $(MPIFC) 2> /dev/null)
	TARGETS=libchord.so
ifneq ($(MPI),)
    $(info MPI Compiler found so compiling MPI polychord)
	TARGETS+=libchord_mpi.so

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

all: $(TARGETS)

ifeq ($(COSMOSIS_DEBUG),1)
FFLAGS += -fPIC -ffree-line-length-none
else
FFLAGS += -O3 -fPIC -ffree-line-length-none
endif

FFLAGS += -fno-stack-arrays

CHORDLIBDIR = ./
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
NOEXECSTACK_LDFLAG = -Wl,-z,noexecstack
else
NOEXECSTACK_LDFLAG =
endif

PCOBJECTS = utils.o abort.o array_utils.o settings.o calculate.o mpi_utils.o random_utils.o chordal_sampling.o run_time_info.o clustering.o params.o priors.o read_write.o feedback.o generate.o ini.o nested_sampling.o interfaces.o 
PCOBJECTS_MPI = $(patsubst %.o,mpi/%.o,$(PCOBJECTS))


%.o: %.f90
	$(FC) $(FFLAGS) -c -o $@ $^ 

%.o: %.F90
	$(FC) $(FFLAGS) -c -o $@ $^ 

mpi/%.o: %.f90 mpi/.phony
	$(MPIFC) $(FFLAGS) -DMPI -c -o $@ $<

mpi/%.o: %.F90 mpi/.phony
	$(MPIFC) $(FFLAGS) -DMPI -c -o $@ $<

mpi/.phony:
	@mkdir -p mpi
	@touch mpi/.phony

test:
	@echo "Alas, PolyChord has no tests"

libchord_mpi.so: $(PCOBJECTS_MPI)
	$(MPIFC) -shared -o  $(CURDIR)/$@ $^ $(LIBS) $(FFLAGS) $(LDFLAGS) $(NOEXECSTACK_LDFLAG)
	
libchord.so: $(PCOBJECTS) 
	$(FC) -shared -o  $(CURDIR)/$@ $^ $(LIBS) $(FFLAGS) $(LDFLAGS) $(NOEXECSTACK_LDFLAG)

clean: 
	-rm -f $(CHORDLIBDIR)/libchord_mpi.*
	-rm -rf mpi
	-rm -f $(CHORDLIBDIR)/libchord.* *.o *.mod

