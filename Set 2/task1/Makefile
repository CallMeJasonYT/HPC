CC=gcc
CFLAGS=-O3 -mavx -fopenmp -ftree-vectorize -march=native
LIBS=-lm
dcache_size := $(shell lscpu | grep "L1d cache" | awk '{print $$3}') # Gets the dcache_size of this computer
main_memory_size = 512 # Change to adjust available memory in MegaBytes for the Stream-Like Benchmark 

all: bench clean

bench: bench.c weno.h 
	@echo "Compiling bench.c...\n";\
	$(CC) $(CFLAGS) -o bench bench.c $(LIBS);\
	echo "Running benchmark...";\
	./bench $(dcache_size) $(main_memory_size)

clean:
	@rm -f bench