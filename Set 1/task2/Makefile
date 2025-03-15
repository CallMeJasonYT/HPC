N_PROCESSES = $(shell python3 -c "import os; print(max(1, os.cpu_count()))") # Change to adjust the number of Processors

all: run_serial task2_a task2_b task2_c clean

run_serial:
	@echo "Available CPU Count: $(N_PROCESSES)";\
	echo "Running serial script...";\
	python3 gs.py;\
	echo "";\
	sleep 1

task2_a:
	@echo "Running gs_pool.py";\
	serial_time=$$(cat serial_time.txt);\
	python3 gs_pool.py $$serial_time $(N_PROCESSES);\
	echo "";\
	sleep 1

task2_b:
	@echo "Running gs_mpifutures.py";\
	serial_time=$$(cat serial_time.txt);\
	mpirun -n $(N_PROCESSES) --use-hwthread-cpus python3 gs_mpifutures.py $$serial_time;\
	echo "";\
	sleep 1
	
task2_c:
	@echo "Running gs_mpiMasterWorker.py";\
	serial_time=$$(cat serial_time.txt);\
	mpirun -n $(N_PROCESSES) --use-hwthread-cpus python3 gs_mpiMasterWorker.py $$serial_time;\
	echo "";\
	sleep 1

clean:
	@echo "Cleaning up..."
	@rm -f serial_time.txt
