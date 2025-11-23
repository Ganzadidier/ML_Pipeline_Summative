"""
MobileNetV2 End-to-End Prediction Module
Uses the trained full model saved as mobilenet_final.keras
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

IMG_SIZE = (224, 224)  # MobileNetV2 default


# -----------------------------------------------------------
# LOAD MODEL
# -----------------------------------------------------------
def load_mobilenet_model(model_path='../notebooks/models/mobilenet_final_tf2.h5'):
    """
    Load the full MobileNetV2 model trained in the notebook.

    Returns:
        model: Keras model ready for inference
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at: {model_path}")

    print(f"\n[LOADING MODEL]")
    print(f" → {model_path}")

    model = load_model(model_path, compile=False)

    # Compile for inference (no effect on predictions)
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print(" ✓ Model loaded successfully")
    print(f"   - Input shape: {model.input_shape}")
    print(f"   - Output shape: {model.output_shape}")
    print(f"   - Classes: 2 (NORMAL, PNEUMONIA)\n")

    return model


# -----------------------------------------------------------
# IMAGE PREPROCESSOR
# -----------------------------------------------------------
def preprocess_image(img_path):
    """
    Loads and preprocesses an image for MobileNetV2.

    Returns:
        np.ndarray of shape (1, 224, 224, 3)
    """
    img = image.load_img(img_path, target_size=IMG_SIZE)
    arr = image.img_to_array(img) / 255.0  # NOTE: Notebook used rescale(1./255)
    return np.expand_dims(arr, axis=0)


# -----------------------------------------------------------
# SINGLE IMAGE PREDICTION
# -----------------------------------------------------------
def predict_single(model, img_path):
    """
    Predict a single image using the full MobileNet model.

    Returns:
        (label, confidence)
    """
    x = preprocess_image(img_path)
    proba = model.predict(x, verbose=0)[0]  # shape (2,)

    pred_id = int(np.argmax(proba))
    confidence = float(proba[pred_id])

    # Label map — same order as training
    label_map = {0: 'NORMAL', 1: 'PNEUMONIA'}
    label = label_map[pred_id]

    return label, confidence


# -----------------------------------------------------------
# BATCH PREDICTION
# -----------------------------------------------------------
def predict_batch(model, img_paths, batch_size=32):
    """
    Predict multiple images efficiently.

    Returns:
        list of (label, confidence)
    """
    results = []

    for i in range(0, len(img_paths), batch_size):
        batch_paths = img_paths[i:i + batch_size]
        batch_images = []

        for p in batch_paths:
            try:
                batch_images.append(preprocess_image(p)[0])
            except:
                results.append((None, 0.0))
                continue

        if not batch_images:
            continue

        batch = np.array(batch_images)
        probas = model.predict(batch, verbose=0)

        label_map = {0: 'NORMAL', 1: 'PNEUMONIA'}

        for proba in probas:
            pred_id = int(np.argmax(proba))
            confidence = float(proba[pred_id])
            label = label_map[pred_id]
            results.append((label, confidence))

    return results


# -----------------------------------------------------------
# DIRECT EXECUTION TEST
# -----------------------------------------------------------
if __name__ == "__main__":
    print("Testing MobileNetV2 Prediction Module...\n")

    try:
        model = load_mobilenet_model()

        sample = "IM-0122-0001.jpeg"
        if os.path.exists(sample):
            label, conf = predict_single(model, sample)
            print(f"Prediction: {label} ({conf:.4f})")
        else:
            print("No test image found at IM-0122-0001.jpeg")

        print("\n✓ Prediction test completed")

    except Exception as e:
        print(f"\n✗ Failed: {e}")
        raise