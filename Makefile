CC = clang
CFLAGS = -Wall -Werror -Wextra -pedantic -std=c99
SRC = kmeans.c
REF_SRC = ref_kmeans.c
OBJ = ${SRC:.c=.o}
REF_OBJ = ${REF_SRC:.c=.o}

.PHONY: all fast faster fastest

all: 

main: slow

slow: ${SRC}

fast: CFLAGS += -O3
fast: ${SRC}

faster: CFLAGS += -Ofast
faster: ${SRC}

fastest: CFLAGS += -march # Compile according to the system's processor architecture
fastest: CFLAGS += -Ofast # The highest level of optimization possible
fastest: CFLAGS += -fno-signed-zeros # Optimizations for FP arith to ignore the sign of zero
fastest: CFLAGS += -freciprocal-math # Allow the reciprocal of a value to be used instead of dividing by the value if this enables optimizations.
fastest: CFLAGS += -ffp-contract=fast # enables floating-point expression contraction such as forming of fused multiply-add operations if the target has native support for them.
fastest: CFLAGS += -menable-no-nans 
fastest: ${SRC}

danger: CFLAGS += -march
danger: CFLAGS += -Ofast
danger: CFLAGS += -fno-signed-zeros
danger: CFLAGS += -freciprocal-math
danger: CFLAGS += -ffp-contract=fast
danger: CFLAGS += -menable-no-nans
danger: CFLAGS += -funroll-loops
fastest: ${SRC}

