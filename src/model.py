import os
import numpy as np
import joblib
import json
from datetime import datetime

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2

import matplotlib.pyplot as plt


# ================================
# CONFIG
# ================================
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_CLASSES = 2

TRAIN_DIR = "data/train"
VAL_DIR = "data/val"
TEST_DIR = "data/test"

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

FEATURES_TRAIN_PATH = os.path.join(MODEL_DIR, "features_train.npz")
FEATURES_VAL_PATH = os.path.join(MODEL_DIR, "features_val.npz")
FEATURES_TEST_PATH = os.path.join(MODEL_DIR, "features_test.npz")

FEATURE_EXTRACTOR_PATH = os.path.join(MODEL_DIR, "mobilenet_base.keras")
SVM_MODEL_PATH = os.path.join(MODEL_DIR, "svm_model.joblib")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.joblib")
RESULTS_JSON = os.path.join(MODEL_DIR, "results.json")
CM_PNG = os.path.join(MODEL_DIR, "confusion_matrix.png")

# Training hyperparams for fine-tuning schedule
HEAD_EPOCHS = 5
FINE_TUNE_EPOCHS = 5
TOTAL_EPOCHS = HEAD_EPOCHS + FINE_TUNE_EPOCHS
INITIAL_LR = 1e-3
FINE_TUNE_LR = 1e-5


# ================================
# UTILITIES
# ================================

def save_npz(path, features, labels, class_indices=None):
    np.savez_compressed(path, features=features, labels=labels, class_indices=class_indices)


def load_npz(path):
    data = np.load(path, allow_pickle=True)
    return data['features'], data['labels'], (data['class_indices'].item() if 'class_indices' in data else None)


# ================================
# DATA AUGMENTATION (Keras layers)
# ================================

def get_augmentation_model():
    return keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.06),  # ~20 degrees
        layers.RandomZoom(0.08),
        layers.RandomTranslation(0.06, 0.06),
        layers.RandomContrast(0.1),
    ], name="data_augmentation")


# ================================
# BUILD BASE (MobileNetV2)
# ================================

def build_base_model(trainable=False):
    base = MobileNetV2(weights='imagenet', include_top=False, pooling='avg', input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base.trainable = trainable
    return base


# ================================
# BUILD CLASSIFIER HEAD (on top of base)
# ================================

def build_full_model(base_model, num_classes=NUM_CLASSES, augmentation=None):
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = inputs
    if augmentation is not None:
        x = augmentation(x)
    x = base_model(x, training=False)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = keras.Model(inputs, outputs)
    return model


# ================================
# DATA GENERATORS (no augmentation for feature extraction)
# ================================

def get_generators(train_dir, val_dir=None, test_dir=None):
    train_datagen = ImageDataGenerator(rescale=1./255)
    val_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='sparse'
    )

    val_gen = None
    if val_dir is not None:
        val_gen = val_datagen.flow_from_directory(
            val_dir,
            target_size=(IMG_SIZE, IMG_SIZE),
            batch_size=BATCH_SIZE,
            class_mode='sparse',
            shuffle=False
        )

    test_gen = None
    if test_dir is not None:
        test_gen = val_datagen.flow_from_directory(
            test_dir,
            target_size=(IMG_SIZE, IMG_SIZE),
            batch_size=BATCH_SIZE,
            class_mode='sparse',
            shuffle=False
        )

    return train_gen, val_gen, test_gen


# ================================
# TRAIN HEAD THEN FINE-TUNE
# ================================

def train_and_fine_tune(train_dir, val_dir):
    augmentation = get_augmentation_model()
    base = build_base_model(trainable=False)
    model = build_full_model(base, augmentation=augmentation)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=INITIAL_LR),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    train_gen, val_gen, _ = get_generators(train_dir, val_dir)

    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-7, verbose=1),
        keras.callbacks.ModelCheckpoint(FEATURE_EXTRACTOR_PATH, monitor='val_loss', save_best_only=True, save_weights_only=False, verbose=1)
    ]

    print(f"\n=== Training head for {HEAD_EPOCHS} epochs ===")
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=HEAD_EPOCHS,
        callbacks=callbacks
    )

    # Unfreeze last N layers of base for fine-tuning
    unfrozen = 50
    print(f"\n=== Unfreezing last {unfrozen} layers and fine-tuning for {FINE_TUNE_EPOCHS} epochs ===")

    # Recreate base model reference used inside full model
    # MobileNet layers are named; identify them and set last N trainable
    base_layers = [layer for layer in model.layers if isinstance(layer, tf.keras.Model) and layer.name.startswith('mobilenet')]
    # fallback: get base by index
    base_model_ref = None
    for layer in model.layers:
        if hasattr(layer, 'name') and 'mobilenetv2' in layer.name.lower():
            base_model_ref = layer
            break
    if base_model_ref is None:
        # try to inspect model.summary to find the base
        base_model_ref = base

    # Set last `unfrozen` layers trainable
    total = len(base_model_ref.layers)
    for i, l in enumerate(base_model_ref.layers):
        l.trainable = True if i >= (total - unfrozen) else False

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=FINE_TUNE_LR),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=FINE_TUNE_EPOCHS,
        callbacks=callbacks
    )

    # Save the entire model (will be used as feature extractor later)
    print(f"\nSaving fine-tuned model to {FEATURE_EXTRACTOR_PATH}")
    model.save(FEATURE_EXTRACTOR_PATH)

    # Determine class indices
    class_indices = train_gen.class_indices

    return model, class_indices


# ================================
# FEATURE EXTRACTION (use base output if model includes head)
# ================================
def extract_and_cache_features(model_path, directory, out_path):
    # If cached, load
    if os.path.exists(out_path):
        print(f"Loading cached features from {out_path}")
        feats, labels, class_indices = load_npz(out_path)
        return feats, labels, class_indices

    # Load model and get the base (feature) model
    model = keras.models.load_model(model_path, compile=False)

    # Try to find a layer that produces the pooled features
    # If model was built with MobileNetV2(include_top=False,pooling='avg'), then we can create a new model
    # mapping inputs -> base_output
    base_output = None
    for layer in model.layers:
        if 'global_average' in layer.name or isinstance(layer, layers.GlobalAveragePooling2D):
            base_output = layer.output
            break
    if base_output is None:
        # If not found, assume the penultimate layer is the feature layer
        base_output = model.layers[-2].output

    feat_model = keras.Model(inputs=model.input, outputs=base_output)

    datagen = ImageDataGenerator(rescale=1./255)
    generator = datagen.flow_from_directory(
        directory,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='sparse',
        shuffle=False
    )

    print(f"Extracting features from {directory} ...")
    features = feat_model.predict(generator, verbose=1)
    labels = generator.classes
    class_indices = generator.class_indices

    save_npz(out_path, features, labels, class_indices)

    return features, labels, class_indices


# ================================
# TRAIN SVM ON EXTRACTED FEATURES
# ================================

def train_svm_on_features(X_train, y_train, X_val, y_val):
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Train SVM (RBF)
    svm = SVC(kernel='rbf', probability=True, class_weight='balanced')
    print("\nTraining SVM on extracted features...")
    svm.fit(X_train_scaled, y_train)

    # Save scaler + svm
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(svm, SVM_MODEL_PATH)
    print(f"Saved scaler ({SCALER_PATH}) and SVM ({SVM_MODEL_PATH})")

    # Validation eval
    preds = svm.predict(X_val_scaled)
    acc = accuracy_score(y_val, preds)
    report = classification_report(y_val, preds, output_dict=True)
    cm = confusion_matrix(y_val, preds)

    # Save results
    results = {
        'trained_at': datetime.now().isoformat(),
        'accuracy': float(acc),
        'classification_report': report,
        'confusion_matrix': cm.tolist()
    }
    with open(RESULTS_JSON, 'w') as f:
        json.dump(results, f, indent=2)

    # Save confusion matrix image
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest')
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    for (i, j), val in np.ndenumerate(cm):
        plt.text(j, i, val, ha='center', va='center', color='white' if val > cm.max()/2 else 'black')
    plt.tight_layout()
    plt.savefig(CM_PNG)
    plt.close()

    print(f"Validation accuracy: {acc:.4f}")
    print("Confusion matrix saved to", CM_PNG)

    return svm, scaler


# ================================
# EVALUATE ON TEST SET
# ================================

def evaluate_on_test(model_path, svm_path, scaler_path, test_dir):
    print("\n=== Evaluating on test set ===")
    X_test, y_test, _ = extract_and_cache_features(model_path, test_dir, FEATURES_TEST_PATH)

    scaler = joblib.load(scaler_path)
    svm = joblib.load(svm_path)

    X_test_scaled = scaler.transform(X_test)
    preds = svm.predict(X_test_scaled)

    print("Accuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds))
    cm = confusion_matrix(y_test, preds)
    print("Confusion Matrix:\n", cm)


# ================================
# MAIN
# ================================

def main(train=True):
    if train:
        model, class_idx = train_and_fine_tune(TRAIN_DIR, VAL_DIR)

        # Extract features (cached)
        X_train, y_train, _ = extract_and_cache_features(FEATURE_EXTRACTOR_PATH, TRAIN_DIR, FEATURES_TRAIN_PATH)
        X_val, y_val, _ = extract_and_cache_features(FEATURE_EXTRACTOR_PATH, VAL_DIR, FEATURES_VAL_PATH)

        # Train SVM on features
        svm, scaler = train_svm_on_features(X_train, y_train, X_val, y_val)

        # Evaluate on test
        if os.path.exists(TEST_DIR):
            evaluate_on_test(FEATURE_EXTRACTOR_PATH, SVM_MODEL_PATH, SCALER_PATH, TEST_DIR)
    else:
        # Only evaluate if models exist
        if not os.path.exists(FEATURE_EXTRACTOR_PATH) or not os.path.exists(SVM_MODEL_PATH) or not os.path.exists(SCALER_PATH):
            raise FileNotFoundError("Missing models/features. Run with training enabled first.")
        evaluate_on_test(FEATURE_EXTRACTOR_PATH, SVM_MODEL_PATH, SCALER_PATH, TEST_DIR)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='Run training (fine-tune + SVM)')
    args = parser.parse_args()

    main(train=args.train)