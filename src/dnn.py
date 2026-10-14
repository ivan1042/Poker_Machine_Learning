import numpy as np
import tensorflow as tf
from tensorflow import keras
from scikeras.wrappers import KerasClassifier

from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, f1_score


class ToDenseFloat32(BaseEstimator, TransformerMixin):
    """Keras needs a dense float32 matrix after sklearn preprocessing."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if sparse.issparse(X):
            X = X.toarray()
        return np.asarray(X, dtype=np.float32)

    def build_keras_model(
            meta,
            hidden_1=32,
            hidden_2=16,
            dropout=0.10,
            learning_rate=0.001,
    ):
        # SciKeras learns this after the preprocessor runs in each CV fold.
        n_features = meta["n_features_in_"]
        n_classes = meta["n_classes_"]  # 3: call, fold, raise

        model = keras.Sequential([
            keras.layers.Input(shape=(n_features,)),
            keras.layers.Dense(hidden_1, activation="relu"),
            keras.layers.Dropout(dropout),

            keras.layers.Dense(hidden_2, activation="relu"),
            keras.layers.Dropout(dropout),

            keras.layers.Dense(n_classes, activation="softmax"),
        ])

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        return model

early_stop = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True,
)

keras_classifier = KerasClassifier(
    model=build_keras_model,
    epochs=100,                 # early stopping usually ends earlier
    batch_size=32,
    validation_split=0.15,      # taken only from each training fold
    callbacks=[early_stop],
    verbose=0,
    random_state=42,
)

