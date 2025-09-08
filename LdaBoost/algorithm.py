import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

class LdaBoost:
    def __init__(self, n_estimators=10, learning_rate=0.1, max_depth=3, random_state=42):
        """
        Initialize the model.

        Parameters
        ----------
        n_estimators : int
            Number of boosting iterations.
        learning_rate : float
            Shrinkage applied to each tree update.
        max_depth : int
            Max depth of each base regression tree.
        random_state : int or None, default=42
            Seed for reproducibility of base regressors.
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state

        self.rng = np.random.RandomState(random_state) if random_state is not None else None

        self.estimators = []      # trees per iteration (one per class)
        self.lda_transforms = []  # LDA transformers per iteration
        self.initial_logit = None
        self.classes_ = None

    def softmax(self, F):
        """Row-wise softmax of logits F (n_samples, n_classes)."""
        expF = np.exp(F - np.max(F, axis=1, keepdims=True))
        return expF / np.sum(expF, axis=1, keepdims=True)

    def fit(self, X, y):
        """Fit the model on features X and multiclass target y."""
        n_samples = X.shape[0]
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        one_hot_y = np.eye(n_classes)[np.searchsorted(self.classes_, y)]

        prior = np.mean(one_hot_y, axis=0)
        prior = np.clip(prior, 1e-5, 1 - 1e-5)
        self.initial_logit = np.log(prior)

        F = np.tile(self.initial_logit, (n_samples, 1))

        for m in range(self.n_estimators):
            if m == 0:
                lda = LinearDiscriminantAnalysis(n_components=None)
                X_lda = lda.fit_transform(X, y)
            else:
                p = self.softmax(F)
                residuals = one_hot_y - p
                labels = np.argmax(residuals, axis=1)
                lda = LinearDiscriminantAnalysis(n_components=None)
                X_lda = lda.fit_transform(X, labels)

            self.lda_transforms.append(lda)

            p = self.softmax(F)
            residuals = one_hot_y - p

            estimators_m = []
            for k in range(n_classes):
                seed = self.rng.randint(0, 10000) if self.rng is not None else None
                reg = DecisionTreeRegressor(max_depth=self.max_depth, random_state=seed)
                reg.fit(X_lda, residuals[:, k])
                estimators_m.append(reg)
                F[:, k] += self.learning_rate * reg.predict(X_lda)

            self.estimators.append(estimators_m)

        return self

    def predict_proba(self, X):
        """Predict class probabilities for X."""
        n_samples = X.shape[0]
        F = np.tile(self.initial_logit, (n_samples, 1))

        for lda, estimators_m in zip(self.lda_transforms, self.estimators):
            X_lda = lda.transform(X)
            for k, reg in enumerate(estimators_m):
                F[:, k] += self.learning_rate * reg.predict(X_lda)

        return self.softmax(F)

    def predict(self, X):
        """Predict class labels for X."""
        proba = self.predict_proba(X)
        class_indices = np.argmax(proba, axis=1)
        return self.classes_[class_indices]

    def cross_validate(self, X, y, cv=10):
        """Stratified K-fold CV; returns accuracy per fold."""
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        accuracies = []

        for train_idx, test_idx in skf.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            model = LdaBoost(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                random_state=self.random_state
            )

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracies.append(accuracy_score(y_test, y_pred))

        return accuracies