"""
prediction.py - HOG + Random Forest predictor (no CLI)
This file is now a pure Python module used by Flask.
"""

import os
import cv2
import numpy as np
from skimage.feature import hog
from joblib import load


class PneumoniaPredictor:
    """Used programmatically by Flask for predictions."""

    def __init__(self, model_path="notebooks/random_forest_hog.pkl"):

        possible_paths = [
            model_path,
            "models/random_forest_hog.pkl",
            "random_forest_hog.pkl",
            "models/random_forest_base.pkl"
        ]

        self.model = None
        for path in possible_paths:
            if os.path.exists(path):
                print(f"[PREDICTOR] Loaded model: {path}")
                self.model = load(path)
                break

        if self.model is None:
            print("[PREDICTOR] Warning: no trained Random Forest model found.")
            print(f"[PREDICTOR] Looked for: {possible_paths}")
            print("[PREDICTOR] The Flask app can still start, but predictions will fail")

        self.class_names = ['NORMAL', 'PNEUMONIA']
        self.img_size = 128

    def _ensure_model(self):
        if self.model is None:
            raise RuntimeError(
                "Random Forest model not loaded. Upload or train a model first."
            )

    def extract_hog_features(self, img_path):
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Could not load image: {img_path}")

        img = cv2.resize(img, (self.img_size, self.img_size))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        return hog(
            gray,
            orientations=12,
            pixels_per_cell=(6, 6),
            cells_per_block=(3, 3),
            block_norm="L2-Hys",
            transform_sqrt=True
        )

    def predict_single(self, image_path):
        """Predict a single uploaded image file."""

        self._ensure_model()
        features = self.extract_hog_features(image_path)

        pred = self.model.predict([features])[0]
        proba = self.model.predict_proba([features])[0]

        return {
            "filename": os.path.basename(image_path),
            "prediction": self.class_names[pred],
            "confidence": float(proba[pred]),
            "probabilities": {
                self.class_names[i]: float(p)
                for i, p in enumerate(proba)
            }
        }
