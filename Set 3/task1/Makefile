matrix_size = 1024 # Define the matrix size as a Makefile variable

all: cudaRun clean

cudaRun:
	@echo "Compiling complex_matrix_multiplication_omp.cu..."; \
	nvcc -Xcompiler "-fopenmp -fno-lto" -arch=sm_70 -o complex_matrix_multiplication_omp complex_matrix_multiplication_omp.cu -DMATRIX_SIZE=$(matrix_size); \
	echo "Running complex_matrix_multiplication_omp with matrix size $(matrix_size)..."; \
	./complex_matrix_multiplication_omp $(matrix_size); 

clean:
	@rm -f complex_matrix_multiplication_omp
