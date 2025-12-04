SUPPORTS_CXX := FALSE
ifeq ($(COMPILER),intel)
  FC_AUTO_R8 :=  -r8 
  MPIFC :=  mpiifort 
  FFLAGS_NOOPT :=  -O0 
  CXX_LDFLAGS :=  -cxxlib 
  SUPPORTS_CXX := TRUE
  FFLAGS :=  -qno-opt-dynamic-align  -convert big_endian -assume byterecl -ftz -traceback -assume realloc_lhs -fp-model source 
  FIXEDFLAGS :=  -fixed -132
  SCC :=  icc 
  SFC :=  ifort 
  MPICC :=  mpiicc  
  MPI_PATH := /public/software/mpi/intelmpi/2021.3.0
  CFLAGS :=  -qno-opt-dynamic-align -fp-model precise -std=gnu99 
  ESMF_LIBDIR := /public/home/cuit_lsh/esmf-ESMF_8_1_1/lib
  MPICXX :=  mpiicpc 
  FREEFLAGS :=  -free 
  CXX_LINKER := FORTRAN
  SCXX := icpc 
endif
CPPDEFS := $(CPPDEFS)  -DCESMCOUPLED 
ifeq ($(MODEL),pop)
  CPPDEFS := $(CPPDEFS)  -D_USE_FLOW_CONTROL 
endif
ifeq ($(MODEL),ufsatm)
  FFLAGS := $(FFLAGS)  $(FC_AUTO_R8) 
  CPPDEFS := $(CPPDEFS)  -DSPMD 
endif
ifeq ($(MODEL),mom)
  FFLAGS := $(FFLAGS)  $(FC_AUTO_R8) -Duse_LARGEFILE
endif
ifeq ($(COMPILER),intel)
  CPPDEFS := $(CPPDEFS) -DFORTRANUNDERSCORE -DCPRINTEL
  SLIBS := $(SLIBS)  -L/public/software/mathlib/netcdf/4.4.1/intel/lib -lnetcdff -lnetcdf
  SLIBS := $(SLIBS)  -L/public/software/mathlib/lapack/intel/3.8.0/lib -llapack 
  SLIBS := $(SLIBS)  -L/public/software/mathlib/BLAS/3.10.0/lib -lblas 
  ifeq ($(compile_threaded),TRUE)
    FFLAGS := $(FFLAGS)  -qopenmp 
    CFLAGS := $(CFLAGS)  -qopenmp 
  endif
  ifeq ($(DEBUG),TRUE)
    FFLAGS := $(FFLAGS)  -O0 -g -check uninit -check bounds -check pointers -fpe0 
    CFLAGS := $(CFLAGS)  -O0 -g 
  endif
  ifeq ($(DEBUG),FALSE)
    FFLAGS := $(FFLAGS)  -O2 -debug minimal 
    CFLAGS := $(CFLAGS)  -O2 -debug minimal 
  endif
  ifeq ($(MPILIB),intelmpi)
    SLIBS := $(SLIBS)  -mkl=cluster 
  endif
  ifeq ($(compile_threaded),TRUE)
    LDFLAGS := $(LDFLAGS)  -qopenmp 
  endif
endif
ifeq ($(MODEL),ufsatm)
  INCLDIR := $(INCLDIR)  -I$(EXEROOT)/atm/obj/FMS 
endif
