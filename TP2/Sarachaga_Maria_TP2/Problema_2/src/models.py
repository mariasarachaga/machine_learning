import numpy as np
from collections import Counter

###########################################################################
def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def one_hot(y, num_classes):
    return np.eye(num_classes)[y]

def logistic_regression(X_train, y_train, X_test, lr=0.01, epochs=1000, reg_lambda=0.0, num_classes=3):
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    n_samples, n_features = X_train.shape
    num_classes = len(np.unique(y_train))
    W = np.zeros((n_features, num_classes))
    b = np.zeros((1, num_classes))
    y_one_hot = one_hot(y_train, num_classes)
    for _ in range(epochs):
        scores = np.dot(X_train, W) + b
        probs = softmax(scores)
        dW = (1/n_samples) * np.dot(X_train.T, (probs - y_one_hot)) + (reg_lambda/n_samples) * W
        db = (1/n_samples) * np.sum(probs - y_one_hot, axis=0, keepdims=True)
        W -= lr * dW
        b -= lr * db
    test_scores = np.dot(X_test, W) + b
    y_pred = np.argmax(softmax(test_scores), axis=1)
    return W, b, y_pred

####################################################################################
def entropy(y):
    values, counts = np.unique(y, return_counts=True)
    probs = counts / counts.sum()
    return -np.sum(probs * np.log2(probs))

def split_dataset(X, y, feature_idx, threshold):
    left_mask = X[:, feature_idx] <= threshold
    right_mask = X[:, feature_idx] > threshold
    return X[left_mask], y[left_mask], X[right_mask], y[right_mask]

def most_common_label(y):
    values, counts = np.unique(y, return_counts=True)
    return values[np.argmax(counts)]

class Node:
    def __init__(self, feature_idx=None, threshold=None, left=None, right=None, value=None):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2, n_features=None):
        # Renombramos para que coincida con la lógica del segundo código
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_split
        self.max_features = n_features

    def entropy(self, labels):
        counts = np.bincount(labels)
        probs = counts / len(labels)
        return -np.sum([p * np.log2(p) for p in probs if p > 0])

    def split_dataset(self, X, y, feature_idx, threshold):
        mask = X[:, feature_idx] <= threshold
        return X[mask], y[mask], X[~mask], y[~mask]

    def select_features_to_use(self, X):
        n_features = X.shape[1]
        if self.max_features is None:
            return list(range(n_features))
        elif isinstance(self.max_features, int):
            return np.random.choice(n_features, size=self.max_features, replace=False)
        else:
            raise ValueError("max_features must be int or None")
    
    def find_best_split(self, X, y):
        best_gain = -1
        best_feature, best_threshold = None, None
        parent_entropy = self.entropy(y)
        feature_indices = self.select_features_to_use(X)

        for feature in feature_indices:
            thresholds = np.unique(X[:, feature])
            for t in thresholds:
                X_left, y_left, X_right, y_right = self.split_dataset(X, y, feature, t)
                if len(y_left) == 0 or len(y_right) == 0:
                    continue
                p_left = len(y_left) / len(y)
                p_right = 1 - p_left
                gain = parent_entropy - (p_left * self.entropy(y_left) + p_right * self.entropy(y_right))
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = t
        return best_feature, best_threshold

    def _grow_tree(self, X, y, depth=0):
        n_samples = len(y)
        num_labels = len(np.unique(y))
        if depth >= self.max_depth or n_samples < self.min_samples_leaf or num_labels == 1:
            leaf_value = Counter(y).most_common(1)[0][0]
            return Node(value=leaf_value)

        feature, threshold = self.find_best_split(X, y)
        if feature is None:
            leaf_value = Counter(y).most_common(1)[0][0]
            return Node(value=leaf_value)

        X_left, y_left, X_right, y_right = self.split_dataset(X, y, feature, threshold)
        left_child = self._grow_tree(X_left, y_left, depth + 1)
        right_child = self._grow_tree(X_right, y_right, depth + 1)
        return Node(feature_idx=feature, threshold=threshold, left=left_child, right=right_child)

    def fit(self, X, y):
        self.root = self._grow_tree(X, y)

    def predict_single(self, x, node=None):
        if node is None:
            node = self.root
        if node.value is not None:
            return node.value
        if x[node.feature_idx] <= node.threshold:
            return self.predict_single(x, node.left)
        else:
            return self.predict_single(x, node.right)

    def predict(self, X):
        return np.array([self.predict_single(x) for x in X])

class RandomForest:
    def __init__(self, n_trees=10, max_depth=None, min_samples_split=2, max_features=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        self.classes_ = np.unique(y)
        n_samples = X.shape[0]
        for _ in range(self.n_trees):
            idxs = np.random.choice(n_samples, n_samples, replace=True)
            tree = DecisionTree(max_depth=self.max_depth,
                                min_samples_split=self.min_samples_split,
                                n_features=self.max_features)
            tree.fit(X[idxs], y[idxs])
            self.trees.append(tree)

    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        y_pred = []
        for i in range(X.shape[0]):
            vals, counts = np.unique(tree_preds[:, i], return_counts=True)
            y_pred.append(vals[np.argmax(counts)])
        return np.array(y_pred)
    
    def predict_proba(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees]) 
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        proba = np.zeros((n_samples, n_classes))

        for i in range(n_samples):
            counts = np.zeros(n_classes)
            for pred in tree_preds[:, i]:
                class_idx = np.where(self.classes_ == pred)[0][0]
                counts[class_idx] += 1
            proba[i] = counts / self.n_trees
        return proba

