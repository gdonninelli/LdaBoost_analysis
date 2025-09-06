import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

class LdaBoost:
    def __init__(self, n_estimators=10, learning_rate=0.1, max_depth=3, random_state=None):
        """
        n_estimators: numero di iterazioni boosting
        learning_rate: tasso di apprendimento
        max_depth: profondità massima dei singoli alberi base
        random_state: seme per la riproducibilità (per i regressori ad albero)
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
        
        # Se random_state è definito, inizializziamo un generatore di numeri casuali
        self.rng = np.random.RandomState(random_state) if random_state is not None else None
        
        self.estimators = []         # Lista per salvare gli alberi per ogni iterazione (uno per ciascuna classe)
        self.lda_transforms = []     # Lista per salvare le trasformazioni LDA ad ogni iterazione
        self.initial_logit = None    # Logit iniziale basato sulle probabilità a priori
        self.classes_ = None         # Classi uniche presenti nel target

    def softmax(self, F):
        """
        Calcola la softmax riga per riga.
        F: matrice di logits di forma (n_samples, n_classes)
        """
        expF = np.exp(F - np.max(F, axis=1, keepdims=True))
        return expF / np.sum(expF, axis=1, keepdims=True)
    
    def fit(self, X, y):
        """
        Allena il modello sui dati X e sul target politomico y.
        """
        n_samples = X.shape[0]
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        
        # Codifica one-hot del target
        one_hot_y = np.eye(n_classes)[np.searchsorted(self.classes_, y)]
        
        # Calcola le probabilità a priori per ciascuna classe e la logit iniziale
        prior = np.mean(one_hot_y, axis=0)
        prior = np.clip(prior, 1e-5, 1-1e-5)
        self.initial_logit = np.log(prior)  # vettore di lunghezza n_classes
        
        # Inizializzazione della matrice F (logits) per ciascun campione: stessa logit iniziale per tutti
        F = np.tile(self.initial_logit, (n_samples, 1))
        
        for m in range(self.n_estimators):
            if m == 0:
                # Prima iterazione: LDA con target originale (multiclasse)
                lda = LinearDiscriminantAnalysis(n_components=None)
                X_lda = lda.fit_transform(X, y)
            else:
                # Calcola le probabilità attuali tramite softmax
                p = self.softmax(F)
                # Pseudo-residui: differenza tra codifica one-hot e probabilità attuali
                residuals = one_hot_y - p  # shape (n_samples, n_classes)
                # Per usare LDA: definiamo per ogni campione l'etichetta del residuo maggiore
                labels = np.argmax(residuals, axis=1)
                lda = LinearDiscriminantAnalysis(n_components=None)
                X_lda = lda.fit_transform(X, labels)
            
            # Salva la trasformazione LDA corrente
            self.lda_transforms.append(lda)
            
            # Ricalcola i residui in base alle logits correnti
            p = self.softmax(F)
            residuals = one_hot_y - p  # shape (n_samples, n_classes)
            
            estimators_m = []
            # Per ciascuna classe si allena un regressore per predire il residuo della classe
            for k in range(n_classes):
                # Genera un seme unico per ogni regressore se random_state è definito
                seed = self.rng.randint(0, 10000) if self.rng is not None else None
                reg = DecisionTreeRegressor(max_depth=self.max_depth, random_state=seed)
                reg.fit(X_lda, residuals[:, k])
                estimators_m.append(reg)
                # Aggiornamento della logit per la classe k
                update = reg.predict(X_lda)
                F[:, k] += self.learning_rate * update
            
            self.estimators.append(estimators_m)
        
        return self

    def predict_proba(self, X):
        """
        Restituisce le probabilità predette per ciascuna classe.
        """
        n_samples = X.shape[0]
        F = np.tile(self.initial_logit, (n_samples, 1))
        
        # Applica in sequenza le trasformazioni e gli aggiornamenti per ogni iterazione
        for lda, estimators_m in zip(self.lda_transforms, self.estimators):
            X_lda = lda.transform(X)
            for k, reg in enumerate(estimators_m):
                F[:, k] += self.learning_rate * reg.predict(X_lda)
        
        return self.softmax(F)

    def predict(self, X):
        """
        Restituisce le classi predette (la classe con la massima probabilità).
        """
        proba = self.predict_proba(X)
        class_indices = np.argmax(proba, axis=1)
        return self.classes_[class_indices]

    def cross_validate(self, X, y, cv=10):
        """
        Esegue una cross-validation stratificata e restituisce l'accuratezza per ogni fold.
        
        Utilizza lo stesso random_state e gli stessi parametri del modello.
        """
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        accuracies = []
        
        for train_idx, test_idx in skf.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Crea una nuova istanza del modello con gli stessi parametri
            model = LdaBoost(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                random_state=self.random_state
            )
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            accuracies.append(acc)
        
        return accuracies
