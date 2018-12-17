CC = gcc
CFLAGS = -Wall -Werror -Wextra -pedantic -std=c99 -fopenmp -mavx2
LFLAGS = -lm
SRC = kmeans.c
REF_SRC = ref_kmeans.c
OBJ = ${SRC:.c=.o}
REF_OBJ = ${REF_SRC:.c=.o}

.PHONY: all fast faster fastest

all: kmeans

kmeans: ${SRC}
	${CC} -o kmeans $^ ${CFLAGS} ${LFLAGS}

time: kmeans
	#time -p ./kmeans 3 20 .01 2351 200000 Xtest.dat output.dat
	time -p ./kmeans 3 2 .01 2351 900000 Xtrain.dat output.dat

eval: time
	#python eval.py output.dat Ytest.dat
	python eval.py output.dat Ytrain.dat

fast: CFLAGS += -O3
fast: kmeans

faster: CFLAGS += -Ofast
faster: kmeans

fastest: CFLAGS += -march=native # Compile according to the system's processor architecture
fastest: CFLAGS += -Ofast # The highest level of optimization possible
fastest: CFLAGS += -fno-signed-zeros # Optimizations for FP arith to ignore the sign of zero
fastest: CFLAGS += -freciprocal-math # Allow the reciprocal of a value to be used instead of dividing by the value if this enables optimizations.
fastest: CFLAGS += -ffp-contract=fast # enables floating-point expression contraction such as forming of fused multiply-add operations if the target has native support for them.
fastest: CFLAGS += -finline-functions # Inlines all "simple functions"
fastest: kmeans

danger: CFLAGS += -march=native
danger: CFLAGS += -Ofast
danger: CFLAGS += -fno-signed-zeros
danger: CFLAGS += -freciprocal-math
danger: CFLAGS += -ffp-contract=fast
danger: CFLAGS += -funroll-loops
danger: CFLAGS += -finline-functions
danger: kmeans

vectorize: CFLAGS += -O2 -ftree-vectorize -msse2 -ftree-vectorizer-verbose=5
vectorize: kmeans

clean:
	${RM} ${OBJ} kmeans
