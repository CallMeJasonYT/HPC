import time
import multiprocessing
import sys
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

pg = ParameterGrid(params)

def evaluate_model(p):
    l1 = p['mlp_layer1']
    l2 = p['mlp_layer2']
    l3 = p['mlp_layer3']
    m = MLPClassifier(hidden_layer_sizes=(l1, l2, l3))
    m.fit(X_train, y_train)
    y_pred = m.predict(X_test)
    ac = accuracy_score(y_pred, y_test)
    return (p, ac)

start_time = time.time()
proc = int(sys.argv[2])

if __name__ == '__main__':
    with multiprocessing.Pool(processes=proc) as pool:
        results = pool.map(evaluate_model, pg)

    for i, (params, ac) in enumerate(results):
        print(f"Parameters: {params}, Accuracy: {ac}")

end_time = time.time()

serial_time = float(sys.argv[1])

parallel_time = end_time - start_time
speedup = serial_time / parallel_time
print(f"Execution Time with Multiprocessing.Pool: {parallel_time:.2f} seconds")
print(f"Speedup with Multiprocessing.Pool: {speedup:.2f}")
