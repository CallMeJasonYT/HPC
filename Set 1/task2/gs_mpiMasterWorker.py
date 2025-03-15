import time
import sys
from mpi4py import MPI
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import ParameterGrid
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=10000, random_state=42, n_features=2, n_informative=2, n_redundant=0, class_sep=0.8)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

params = [{'mlp_layer1': [16, 32],
           'mlp_layer2': [16, 32],
           'mlp_layer3': [16, 32]}]

pg = list(ParameterGrid(params))

def evaluate_model(p):
    l1 = p['mlp_layer1']
    l2 = p['mlp_layer2']
    l3 = p['mlp_layer3']
    m = MLPClassifier(hidden_layer_sizes=(l1, l2, l3))
    m.fit(X_train, y_train)
    y_pred = m.predict(X_test)
    ac = accuracy_score(y_pred, y_test)
    return (p, ac)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if rank == 0:
    start_time = time.time()
    chunk_size = len(pg) // (size - 1)
    for i in range(1, size):
        comm.send(pg[(i-1)*chunk_size:i*chunk_size], dest=i)

    results = []
    for i in range(1, size):
        results += comm.recv(source=i)

    for i, (params, ac) in enumerate(results):
        print(f"Parameters: {params}, Accuracy: {ac}")

    end_time = time.time()

    serial_time = float(sys.argv[1])

    parallel_time = end_time - start_time
    speedup = serial_time / parallel_time
    print(f"Execution Time with MPI.Master-Worker: {parallel_time:.2f} seconds")
    print(f"Speedup with MPI.Master-Worker: {speedup:.2f}")

else:
    tasks = comm.recv(source=0)
    results = [evaluate_model(p) for p in tasks]
    comm.send(results, dest=0)
