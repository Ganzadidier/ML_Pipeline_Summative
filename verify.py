"""
Complete Setup Verification Script
Run this to verify your entire ML pipeline is ready
"""

import os
import sys


def check_color(condition):
    """Return colored checkmark or X"""
    return "✓" if condition else "✗"


def verify_setup():
    """Verify complete project setup"""

    print("=" * 70)
    print("ML PIPELINE SETUP VERIFICATION")
    print("=" * 70)
    print()

    all_good = True

    # 1. Check directories
    print("📁 DIRECTORY STRUCTURE")
    print("-" * 70)

    required_dirs = [
        'data',
        'data/train',
        'data/val',
        'data/test',
        'data/retrain',
        'models',
        'src',
        'templates',
        'static/uploads'
    ]

    for dir_path in required_dirs:
        exists = os.path.exists(dir_path)
        print(f"{check_color(exists)} {dir_path}")
        if not exists:
            all_good = False
            print(f"   → Create with: mkdir {dir_path}")

    print()

    # 2. Check model files
    print("🤖 MODEL FILES")
    print("-" * 70)

    model_files = [
        ('models/mobilenet_base.keras', 'MobileNet feature extractor', True),
        ('models/svm_model.joblib', 'SVM classifier', True),
        ('models/scaler.joblib', 'Feature scaler', True),
        ('models/features_train.npz', 'Training features cache', False),
        ('models/features_val.npz', 'Validation features cache', False),
        ('models/features_test.npz', 'Test features cache', False),
    ]

    for file_path, description, required in model_files:
        exists = os.path.exists(file_path)
        status = check_color(exists)
        req_text = " (REQUIRED)" if required else " (optional)"
        print(f"{status} {file_path}{req_text}")
        print(f"   {description}")

        if required and not exists:
            all_good = False
            print(f"   → Generate with: python src/model.py --train")

    print()

    # 3. Check source files
    print("📝 SOURCE CODE FILES")
    print("-" * 70)

    source_files = [
        'src/model.py',
        'src/prediction.py',
        'src/retraining.py',
        'src/preprocessing.py',
        'src/data_acquisition.py',
        'app.py',
        'locustfile.py'
    ]

    for file_path in source_files:
        exists = os.path.exists(file_path)
        print(f"{check_color(exists)} {file_path}")
        if not exists:
            all_good = False

    print()

    # 4. Check templates
    print("🎨 TEMPLATE FILES")
    print("-" * 70)

    template_files = [
        'templates/index.html',
        'templates/predict.html',
        'templates/visualize.html',
        'templates/retrain.html'
    ]

    for file_path in template_files:
        exists = os.path.exists(file_path)
        print(f"{check_color(exists)} {file_path}")
        if not exists:
            all_good = False

    print()

    # 5. Check Python imports
    print("🐍 PYTHON DEPENDENCIES")
    print("-" * 70)

    dependencies = [
        ('tensorflow', 'TensorFlow'),
        ('numpy', 'NumPy'),
        ('PIL', 'Pillow'),
        ('flask', 'Flask'),
        ('sklearn', 'scikit-learn'),
        ('joblib', 'joblib'),
        ('cv2', 'OpenCV')
    ]

    for module, name in dependencies:
        try:
            __import__(module)
            print(f"✓ {name}")
        except ImportError:
            print(f"✗ {name} - Install with: pip install {name.lower()}")
            all_good = False

    print()

    # 6. Test model loading
    print("🧪 MODEL LOADING TEST")
    print("-" * 70)

    try:
        from src.prediction import load_models
        print("Attempting to load models...")
        feature_extractor, svm_model, scaler = load_models()
        print("✓ All models loaded successfully!")
        print(f"  - Feature extractor: {type(feature_extractor).__name__}")
        print(f"  - SVM model: {type(svm_model).__name__}")
        print(f"  - Scaler: {type(scaler).__name__}")
    except FileNotFoundError as e:
        print(f"✗ Model files not found: {e}")
        print("  → Run: python src/model.py --train")
        all_good = False
    except Exception as e:
        print(f"✗ Error loading models: {e}")
        all_good = False

    print()

    # 7. Test database initialization
    print("💾 DATABASE TEST")
    print("-" * 70)

    try:
        from src.retraining import init_database, get_database_statistics
        print("Initializing database...")
        init_database()

        stats = get_database_statistics()
        print(f"✓ Database operational")
        print(f"  - Total uploaded: {stats.get('total_uploaded', 0)}")
        print(f"  - Preprocessed: {stats.get('preprocessed', 0)}")
        print(f"  - Database location: data/retraining.db")
    except Exception as e:
        print(f"✗ Database error: {e}")
        all_good = False

    print()

    # 8. Check data availability
    print("📊 DATASET STATUS")
    print("-" * 70)

    data_dirs = ['data/train', 'data/val', 'data/test']
    classes = ['NORMAL', 'PNEUMONIA']

    has_data = False
    for data_dir in data_dirs:
        for cls in classes:
            cls_path = os.path.join(data_dir, cls)
            if os.path.exists(cls_path):
                count = len([f for f in os.listdir(cls_path)
                             if f.endswith(('.jpg', '.jpeg', '.png'))])
                if count > 0:
                    has_data = True
                    print(f"✓ {data_dir}/{cls}: {count} images")
                else:
                    print(f"⚠ {data_dir}/{cls}: empty")

    if not has_data:
        print()
        print("⚠ No training data found!")
        print("  → Download dataset with: python download_dataset.py")
        all_good = False

    print()

    # Final summary
    print("=" * 70)
    if all_good:
        print("✅ ALL CHECKS PASSED - YOU'RE READY!")
        print("=" * 70)
        print()
        print("Next steps:")
        print("1. Start application: python app.py")
        print("2. Open browser: http://localhost:5000")
        print("3. Test prediction and retraining")
        print("4. Record video demo")
        print()
        return 0
    else:
        print("❌ SOME CHECKS FAILED - FIX ISSUES ABOVE")
        print("=" * 70)
        print()
        print("Common fixes:")
        print("1. Create missing directories: mkdir data models templates")
        print("2. Train model: python src/model.py --train")
        print("3. Install dependencies: pip install -r requirements.txt")
        print("4. Download dataset: python download_dataset.py")
        print()
        return 1


if __name__ == "__main__":
    exit_code = verify_setup()
    sys.exit(exit_code)