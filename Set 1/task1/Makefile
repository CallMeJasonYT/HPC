N_PROCESSORS_1 = 8 # Change to adjust the number of Processors
N_PROCESSORS_2 = 2 # Change to adjust the number of Processors

all: task1_a task1_b task1_c task1_d

task1_a:
	@echo "Compiling mpi_exscan_pt2.c...";\
	mpicc -fopenmp -o mpi_exscan_pt2  mpi_exscan_pt2.c;\
	echo "Running mpi_exscan_pt2...";\
	mpirun -np $(N_PROCESSORS_1) --use-hwthread-cpus ./mpi_exscan_pt2;\
	echo "";\
	sleep 1

task1_b:
	@echo "Compiling mpi_exscan_omp.c...";\
	mpicc -fopenmp -o mpi_exscan_omp  mpi_exscan_omp.c;\
	echo "Running mpi_exscan_omp...";\
	mpirun -np $(N_PROCESSORS_1) --use-hwthread-cpus ./mpi_exscan_omp;\
	echo "";\
	sleep 1

task1_c:
	@echo "Compiling mpi_exscan_omp_file.c";\
	mpicc -fopenmp -o mpi_exscan_omp_file  mpi_exscan_omp_file.c;\
	echo "Running mpi_exscan_omp_file";\
	mpirun -np $(N_PROCESSORS_2) --use-hwthread-cpus ./mpi_exscan_omp_file;\
	echo "";\
	sleep 1

task1_d:
	@echo "Compiling mpi_exscan_omp_file_compress.c";\
	mpicc -fopenmp -o mpi_exscan_omp_file_compress  mpi_exscan_omp_file_compress.c -lz;\
	echo "Running mpi_exscan_omp_file_compress";\
	mpirun -np $(N_PROCESSORS_2) --use-hwthread-cpus ./mpi_exscan_omp_file_compress;\
	echo "";\
	sleep 1