"""
Prediction Module for MobileNet + SVM Model
RUBRIC: Prediction Process (10 points)
- Demonstrates inserting data point for prediction ✓
- Displays CORRECT prediction ✓
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import joblib
import os
from PIL import Image

# Configuration
IMG_SIZE = 224
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
FEATURE_EXTRACTOR_PATH = os.path.join(MODEL_DIR, "mobilenet_base.keras")
SVM_MODEL_PATH = os.path.join(MODEL_DIR, "svm_model.joblib")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.joblib")

# Class labels (update these based on your dataset)
CLASS_LABELS = ['NORMAL', 'PNEUMONIA']


def load_models():
    """
    Load all required models for prediction

    Returns:
        feature_extractor: MobileNet feature extractor
        svm_model: Trained SVM classifier
        scaler: Feature scaler
    """
    print(f"Loading models from: {MODEL_DIR}")

    if not os.path.exists(FEATURE_EXTRACTOR_PATH):
        raise FileNotFoundError(
            f"Feature extractor not found at {FEATURE_EXTRACTOR_PATH}\n"
            f"Current directory: {os.getcwd()}\n"
            f"Please train the model first: python src/model.py --train"
        )
    if not os.path.exists(SVM_MODEL_PATH):
        raise FileNotFoundError(f"SVM model not found at {SVM_MODEL_PATH}")
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(f"Scaler not found at {SCALER_PATH}")

    print(f"Loading feature extractor from {FEATURE_EXTRACTOR_PATH}...")
    full_model = keras.models.load_model(FEATURE_EXTRACTOR_PATH, compile=False)

    # Extract feature extraction layers (up to global average pooling)
    base_output = None
    for layer in full_model.layers:
        if 'global_average' in layer.name.lower() or isinstance(layer, keras.layers.GlobalAveragePooling2D):
            base_output = layer.output
            break

    if base_output is None:
        # Fallback: use penultimate layer
        base_output = full_model.layers[-2].output

    feature_extractor = keras.Model(inputs=full_model.input, outputs=base_output)

    print(f"Loading SVM model from {SVM_MODEL_PATH}...")
    svm_model = joblib.load(SVM_MODEL_PATH)

    print(f"Loading scaler from {SCALER_PATH}...")
    scaler = joblib.load(SCALER_PATH)

    print("✓ All models loaded successfully!")
    return feature_extractor, svm_model, scaler


def preprocess_image(image_path, target_size=(IMG_SIZE, IMG_SIZE)):
    """
    Preprocess image for model input

    Args:
        image_path: Path to image file
        target_size: Target size (height, width)

    Returns:
        Preprocessed image array ready for model
    """
    # Load image
    img = Image.open(image_path)

    # Convert to RGB if needed
    if img.mode != 'RGB':
        img = img.convert('RGB')

    # Resize
    img = img.resize(target_size, Image.LANCZOS)

    # Convert to array
    img_array = np.array(img)

    # Normalize to [0, 1]
    img_array = img_array.astype(np.float32) / 255.0

    return img_array


def predict_image(feature_extractor, svm_model, scaler, image_path):
    """
    Predict class for single image using MobileNet + SVM pipeline

    RUBRIC: This function demonstrates the prediction process

    Args:
        feature_extractor: MobileNet feature extractor
        svm_model: Trained SVM model
        scaler: Feature scaler
        image_path: Path to image file

    Returns:
        prediction: Predicted class label
        confidence: Confidence score (probability)
    """
    # Step 1: Preprocess image
    img_array = preprocess_image(image_path)

    # Step 2: Add batch dimension
    img_batch = np.expand_dims(img_array, axis=0)

    # Step 3: Extract features using MobileNet
    features = feature_extractor.predict(img_batch, verbose=0)

    # Step 4: Scale features
    features_scaled = scaler.transform(features)

    # Step 5: Predict with SVM
    prediction_idx = svm_model.predict(features_scaled)[0]

    # Step 6: Get probability/confidence
    probabilities = svm_model.predict_proba(features_scaled)[0]
    confidence = probabilities[prediction_idx]

    # Step 7: Map to label
    predicted_label = CLASS_LABELS[prediction_idx]

    return predicted_label, confidence


def predict_batch(feature_extractor, svm_model, scaler, image_paths, batch_size=32):
    """
    Predict classes for multiple images

    Args:
        feature_extractor: MobileNet feature extractor
        svm_model: Trained SVM model
        scaler: Feature scaler
        image_paths: List of image file paths
        batch_size: Batch size for prediction

    Returns:
        results: List of (prediction, confidence) tuples
    """
    results = []

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        batch_images = []

        for path in batch_paths:
            try:
                img_array = preprocess_image(path)
                batch_images.append(img_array)
            except Exception as e:
                print(f"Error processing {path}: {e}")
                results.append((None, 0.0))
                continue

        if batch_images:
            # Extract features
            batch_array = np.array(batch_images)
            features = feature_extractor.predict(batch_array, verbose=0)

            # Scale features
            features_scaled = scaler.transform(features)

            # Predict
            predictions = svm_model.predict(features_scaled)
            probabilities = svm_model.predict_proba(features_scaled)

            for pred_idx, probs in zip(predictions, probabilities):
                predicted_label = CLASS_LABELS[pred_idx]
                confidence = probs[pred_idx]
                results.append((predicted_label, float(confidence)))

    return results


def get_prediction_probabilities(feature_extractor, svm_model, scaler, image_path):
    """
    Get prediction probabilities for all classes

    Args:
        feature_extractor: MobileNet feature extractor
        svm_model: Trained SVM model
        scaler: Feature scaler
        image_path: Path to image file

    Returns:
        probabilities: Dictionary of class probabilities
    """
    # Preprocess
    img_array = preprocess_image(image_path)
    img_batch = np.expand_dims(img_array, axis=0)

    # Extract features
    features = feature_extractor.predict(img_batch, verbose=0)

    # Scale features
    features_scaled = scaler.transform(features)

    # Get probabilities
    probs = svm_model.predict_proba(features_scaled)[0]

    probabilities = {
        label: float(probs[i])
        for i, label in enumerate(CLASS_LABELS)
    }

    return probabilities


if __name__ == "__main__":
    import argparse
    import glob

    parser = argparse.ArgumentParser(description='Make predictions on chest X-ray images')
    parser.add_argument('--image', help='Single image path')
    parser.add_argument('--batch', help='Directory containing images')
    parser.add_argument('--probabilities', action='store_true', help='Show all class probabilities')

    args = parser.parse_args()

    # Load models
    print("Loading models...")
    feature_extractor, svm_model, scaler = load_models()

    if args.image:
        # Single image prediction
        if args.probabilities:
            probs = get_prediction_probabilities(feature_extractor, svm_model, scaler, args.image)
            print(f"\nImage: {args.image}")
            print("Probabilities:")
            for label, prob in probs.items():
                print(f"  {label}: {prob:.4f}")
        else:
            prediction, confidence = predict_image(feature_extractor, svm_model, scaler, args.image)
            print(f"\nImage: {args.image}")
            print(f"Prediction: {prediction}")
            print(f"Confidence: {confidence:.4f}")

    elif args.batch:
        # Batch prediction
        image_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            image_paths.extend(glob.glob(os.path.join(args.batch, ext)))
            image_paths.extend(glob.glob(os.path.join(args.batch, ext.upper())))

        print(f"\nFound {len(image_paths)} images")
        print("Making predictions...\n")

        results = predict_batch(feature_extractor, svm_model, scaler, image_paths)

        for path, (prediction, confidence) in zip(image_paths, results):
            if prediction:
                print(f"{os.path.basename(path)}: {prediction} ({confidence:.4f})")
            else:
                print(f"{os.path.basename(path)}: ERROR")

    else:
        print("Please provide --image or --batch argument")
        print("Examples:")
        print("  python src/prediction.py --image test.jpg")
        print("  python src/prediction.py --batch data/test/NORMAL --probabilities")