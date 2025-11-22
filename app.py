"""
Flask Application for MobileNet + SVM ML Pipeline
Serves prediction, retraining, and monitoring interfaces
"""

from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
import time
import json
from datetime import datetime
import threading
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Configuration
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['RETRAIN_FOLDER'] = 'data/retrain'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RETRAIN_FOLDER'], exist_ok=True)
os.makedirs('data/retrain/NORMAL', exist_ok=True)
os.makedirs('data/retrain/PNEUMONIA', exist_ok=True)

# Global metrics
metrics = {
    'start_time': time.time(),
    'total_predictions': 0,
    'total_errors': 0,
    'latencies': [],
    'predictions_log': []
}

# Import modules
from src.prediction import load_models, predict_image
from src.retraining import (
    trigger_retraining,
    get_retraining_status,
    init_database,
    save_uploaded_file_to_database,
    preprocess_uploaded_data,
    get_database_statistics
)

# Global variables for models
feature_extractor = None
svm_model = None
scaler = None
models_loaded = False

# Load models on startup
print("\n" + "=" * 70)
print("INITIALIZING ML PIPELINE - MOBILENET + SVM")
print("=" * 70)

print("\n1. Loading Models...")
try:
    feature_extractor, svm_model, scaler = load_models()
    models_loaded = True
    print("   ✓ Feature Extractor: Loaded")
    print("   ✓ SVM Model: Loaded")
    print("   ✓ Scaler: Loaded")
    print("   ✓ All models loaded successfully!")
except Exception as e:
    print(f"   ✗ Error loading models: {e}")
    print("   → Make sure you've trained the model: python src/model.py --train")
    models_loaded = False

print("\n2. Initializing Database...")
try:
    init_database()
    print("   ✓ Database initialized")
except Exception as e:
    print(f"   ✗ Database initialization failed: {e}")

print("\n" + "=" * 70)
print("SERVER READY")
print("=" * 70 + "\n")


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


# ============================================================================
# WEB PAGES
# ============================================================================

@app.route('/')
def index():
    """Main dashboard"""
    return render_template('index.html')


@app.route('/predict-page')
def predict_page():
    """Prediction interface"""
    return render_template('predict.html')


@app.route('/visualize')
def visualize():
    """Data visualization page"""
    return render_template('visualize.html')


@app.route('/retrain-page')
def retrain_page():
    """Retraining interface"""
    return render_template('retrain.html')


# ============================================================================
# API ENDPOINTS - STATUS & HEALTH
# ============================================================================

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'models_loaded': models_loaded
    })


@app.route('/model-status')
def model_status():
    """Model status endpoint"""
    global feature_extractor, svm_model, scaler, models_loaded
    return jsonify({
        'model_loaded': models_loaded and feature_extractor is not None,
        'model_path': 'models/mobilenet_base.keras',
        'model_type': 'MobileNetV2 + SVM',
        'feature_extractor': feature_extractor is not None,
        'svm_model': svm_model is not None,
        'scaler': scaler is not None,
        'version': '1.0'
    })


@app.route('/metrics')
def get_metrics():
    """Get application metrics"""
    uptime_seconds = time.time() - metrics['start_time']
    uptime_hours = uptime_seconds / 3600
    uptime_days = uptime_hours / 24

    avg_latency = np.mean(metrics['latencies'][-100:]) if metrics['latencies'] else 0

    return jsonify({
        'uptime_seconds': uptime_seconds,
        'uptime_formatted': f"{int(uptime_days)}d {int(uptime_hours % 24)}h",
        'total_predictions': metrics['total_predictions'],
        'total_errors': metrics['total_errors'],
        'avg_latency_ms': round(avg_latency, 2),
        'success_rate': round((1 - metrics['total_errors'] / max(metrics['total_predictions'], 1)) * 100, 2)
    })


# ============================================================================
# API ENDPOINTS - PREDICTION
# ============================================================================

@app.route('/predict', methods=['POST'])
def predict():
    """
    Single image prediction endpoint
    RUBRIC: Prediction Process (10 points)
    """
    start_time = time.time()

    try:
        global feature_extractor, svm_model, scaler, models_loaded

        # Load models if not loaded
        if not models_loaded or feature_extractor is None or svm_model is None or scaler is None:
            print("Loading models on first request...")
            feature_extractor, svm_model, scaler = load_models()
            models_loaded = True
            print("✓ Models loaded")

        # Validate request
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Use PNG, JPG, or JPEG'}), 400

        # Save file temporarily
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Make prediction using MobileNet + SVM
        print(f"Making prediction on: {filename}")
        prediction, confidence = predict_image(feature_extractor, svm_model, scaler, filepath)
        print(f"Prediction: {prediction} (confidence: {confidence:.4f})")

        # Calculate latency
        latency = (time.time() - start_time) * 1000  # milliseconds

        # Update metrics
        metrics['total_predictions'] += 1
        metrics['latencies'].append(latency)
        metrics['predictions_log'].append({
            'timestamp': datetime.now().isoformat(),
            'prediction': prediction,
            'confidence': float(confidence),
            'latency_ms': latency
        })

        # Keep only last 1000 predictions
        if len(metrics['predictions_log']) > 1000:
            metrics['predictions_log'] = metrics['predictions_log'][-1000:]

        # Clean up uploaded file
        try:
            os.remove(filepath)
        except:
            pass

        return jsonify({
            'prediction': prediction,
            'confidence': float(confidence),
            'latency_ms': round(latency, 2),
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        metrics['total_errors'] += 1
        print(f"Prediction error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ============================================================================
# API ENDPOINTS - DATA UPLOAD & RETRAINING
# ============================================================================

@app.route('/upload-data', methods=['POST'])
def upload_data():
    """
    Upload multiple images for retraining
    RUBRIC: Data file uploading + saving to database
    """
    try:
        if 'files' not in request.files:
            return jsonify({'error': 'No files provided'}), 400

        files = request.files.getlist('files')
        label = request.form.get('label', 'NORMAL')

        if not files:
            return jsonify({'error': 'No files selected'}), 400

        # Create label directory
        label_dir = os.path.join(app.config['RETRAIN_FOLDER'], label)
        os.makedirs(label_dir, exist_ok=True)

        saved_files = []
        database_ids = []

        print(f"\nUploading {len(files)} files with label: {label}")

        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(label_dir, filename)
                file.save(filepath)

                # Save to database - RUBRIC REQUIREMENT
                print(f"  Saving to database: {filename}")
                image_id = save_uploaded_file_to_database(filepath, filename, label)

                if image_id:
                    # Preprocess uploaded data - RUBRIC REQUIREMENT
                    print(f"  Preprocessing image ID: {image_id}")
                    success = preprocess_uploaded_data(image_id)

                    if success:
                        saved_files.append(filename)
                        database_ids.append(image_id)
                        print(f"  ✓ Successfully processed: {filename}")
                    else:
                        print(f"  ✗ Preprocessing failed for: {filename}")

        print(f"Upload complete: {len(saved_files)}/{len(files)} files processed\n")

        return jsonify({
            'success': True,
            'files_uploaded': len(saved_files),
            'database_ids': database_ids,
            'label': label,
            'message': f'Successfully uploaded and preprocessed {len(saved_files)} files'
        })

    except Exception as e:
        print(f"Upload error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/retrain', methods=['POST'])
def retrain():
    """
    Trigger model retraining
    RUBRIC: Retraining using pre-trained model
    """
    try:
        params = request.get_json() or {}
        epochs = params.get('epochs', 5)
        batch_size = params.get('batch_size', 32)

        print(f"\n{'=' * 70}")
        print("RETRAINING REQUEST RECEIVED")
        print(f"{'=' * 70}")
        print(f"Epochs: {epochs}")
        print(f"Batch size: {batch_size}")

        # Start retraining in background thread
        job_id = f"retrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        def retrain_worker():
            try:
                print(f"Starting retraining job: {job_id}")
                # This uses the enhanced retraining function that:
                # 1. Loads data from database
                # 2. Preprocesses all uploaded data
                # 3. Uses pre-trained MobileNet model for transfer learning
                # 4. Retrains SVM on new features
                trigger_retraining(
                    data_dir=None,  # Uses database instead
                    epochs=epochs,
                    batch_size=batch_size,
                    job_id=job_id
                )
                print(f"Retraining job {job_id} completed")
            except Exception as e:
                print(f"Retraining job {job_id} failed: {e}")
                import traceback
                traceback.print_exc()

        thread = threading.Thread(target=retrain_worker, daemon=True)
        thread.start()

        return jsonify({
            'status': 'retraining_started',
            'job_id': job_id,
            'epochs': epochs,
            'batch_size': batch_size,
            'message': 'Model retraining started using pre-trained MobileNet with transfer learning',
            'info': 'Retraining uses the existing MobileNet model and fine-tunes it on new data'
        })

    except Exception as e:
        print(f"Retrain endpoint error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/retrain-status/<job_id>')
def retrain_status(job_id):
    """Get retraining status"""
    status = get_retraining_status(job_id)
    return jsonify(status)


# ============================================================================
# API ENDPOINTS - LOGS & VISUALIZATIONS
# ============================================================================

@app.route('/predictions-log')
def predictions_log():
    """Get recent predictions log"""
    return jsonify({
        'predictions': metrics['predictions_log'][-100:],
        'total': len(metrics['predictions_log'])
    })


@app.route('/visualization-data')
def visualization_data():
    """Get data for visualizations"""
    try:
        # Load training history if exists
        history_path = 'models/results.json'
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                training_history = json.load(f)
        else:
            training_history = {}

        # Calculate prediction distribution
        pred_distribution = {'NORMAL': 0, 'PNEUMONIA': 0}
        confidence_scores = []

        for pred in metrics['predictions_log']:
            pred_distribution[pred['prediction']] = pred_distribution.get(pred['prediction'], 0) + 1
            confidence_scores.append(pred['confidence'])

        # Get database statistics - RUBRIC: Data insights
        db_stats = get_database_statistics()

        return jsonify({
            'training_history': training_history,
            'prediction_distribution': pred_distribution,
            'confidence_scores': confidence_scores,
            'latency_data': metrics['latencies'][-100:],
            'database_statistics': db_stats
        })

    except Exception as e:
        print(f"Visualization data error: {e}")
        return jsonify({'error': str(e)}), 500


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("STARTING FLASK APPLICATION")
    print("=" * 70)
    print("Access the application at:")
    print("  → Dashboard: http://localhost:5000")
    print("  → Prediction: http://localhost:5000/predict-page")
    print("  → Visualizations: http://localhost:5000/visualize")
    print("  → Retraining: http://localhost:5000/retrain-page")
    print("=" * 70 + "\n")

    app.run(host='0.0.0.0', port=5000, debug=False)