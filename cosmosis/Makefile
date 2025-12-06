ifneq (clean,$(MAKECMDGOALS))
include ${COSMOSIS_SRC_DIR}/config/compilers.mk
endif

$(info ${COSMOSIS_SRC_DIR})

include ${COSMOSIS_SRC_DIR}/config/subdirs.mk

SUBDIRS=datablock samplers

ifneq (${COSMOSIS_JULIA_FLAGS},)
$(info Including Julia modules in build)
SUBDIRS+=runtime/julia_modules
endif