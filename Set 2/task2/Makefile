matrix_size = 512 # Change the matrix size

all: cudaRun clean

cudaRun: @echo "Compiling complex_matrix_multiplication.c...";\
	nvcc - arch=compute_70 -code=sm_70,compute_70 -o complex_matrix_multiplication complex_matrix_multiplication.cu;\
	echo "Running complex_matrix_multiplication...";\
	./complex_matrix_multiplication $(matrix_size)

clean:
	@rm -f complex_matrix_multiplication