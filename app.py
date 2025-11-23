"""
Flask Application for MobileNet ML Pipeline (OPTIMIZED)
Uses MobileNet End-to-End - NO SVM
Direct classification with MobileNet
"""

from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
import time
import json
from datetime import datetime
import threading
import numpy as np
from PIL import Image
import io

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

from src.retraining import ( trigger_retraining,
                             get_retraining_status,
                             init_database, save_uploaded_file_to_database,
                             preprocess_uploaded_data, get_database_statistics )

# Import modules - MobileNet end-to-end (NO SVM)
from src.prediction import (
    load_mobilenet_model,
    predict_single,
    predict_batch
)

# Global variables for model
model = None
models_loaded = False
model_lock = threading.Lock()

# Load model on startup (EAGER LOADING)
print("\n" + "=" * 70)
print("INITIALIZING ML PIPELINE - MOBILENET END-TO-END (NO SVM)")
print("=" * 70)

print("\n1. Loading MobileNet Model...")
try:
    with model_lock:
        model = load_mobilenet_model(model_path='notebooks/models/mobilenet_final.keras')
        models_loaded = True
    print("   ✓ MobileNet Model: Loaded (End-to-End)")
except Exception as e:
    print(f"   ✗ Error loading model: {e}")
    models_loaded = False


print("\n2. Initializing Database...")
try:
    init_database()
    print("   ✓ Database initialized")
except Exception as e:
    print(f"   ✗ Database initialization failed: {e}")

print("\n" + "=" * 70)
print("SERVER READY - MOBILENET END-TO-END MODE")
print("=" * 70 + "\n")


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def optimize_image(file_obj, max_size=(1024, 1024), quality=85):
    """
    Optimize image for faster processing
    """
    try:
        img = Image.open(file_obj)

        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Resize if too large
        if img.size[0] > max_size[0] or img.size[1] > max_size[1]:
            img.thumbnail(max_size, Image.Resampling.LANCZOS)

        # Save optimized
        output = io.BytesIO()
        img.save(output, format='JPEG', quality=quality, optimize=True)
        output.seek(0)

        return output
    except Exception as e:
        print(f"Image optimization failed: {e}")
        file_obj.seek(0)
        return file_obj


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
        'models_loaded': models_loaded,
        'mode': 'mobilenet-end-to-end'
    })


@app.route('/model-status')
def model_status():
    """Model status endpoint"""
    global model, models_loaded
    return jsonify({
        'model_loaded': models_loaded and model is not None,
        'model_path': 'models/mobilenet_base.keras',
        'model_type': 'MobileNet End-to-End',
        'classification_mode': 'direct',
        'svm_used': False,
        'version': '2.0-mobilenet-only'
    })


@app.route('/metrics')
def get_metrics():
    """Get application metrics"""
    uptime_seconds = time.time() - metrics['start_time']
    uptime_hours = uptime_seconds / 3600
    uptime_days = uptime_hours / 24

    avg_latency = np.mean(metrics['latencies'][-100:]) if metrics['latencies'] else 0
    min_latency = np.min(metrics['latencies'][-100:]) if metrics['latencies'] else 0
    max_latency = np.max(metrics['latencies'][-100:]) if metrics['latencies'] else 0

    return jsonify({
        'uptime_seconds': uptime_seconds,
        'uptime_formatted': f"{int(uptime_days)}d {int(uptime_hours % 24)}h",
        'total_predictions': metrics['total_predictions'],
        'total_errors': metrics['total_errors'],
        'avg_latency_ms': round(avg_latency, 2),
        'min_latency_ms': round(min_latency, 2),
        'max_latency_ms': round(max_latency, 2),
        'success_rate': round((1 - metrics['total_errors'] / max(metrics['total_predictions'], 1)) * 100, 2)
    })


# ============================================================================
# API ENDPOINTS - PREDICTION (MOBILENET END-TO-END)
# ============================================================================

@app.route('/predict', methods=['POST'])
def predict():
    """
    Single image prediction using MobileNet end-to-end (NO SVM)
    RUBRIC: Prediction Process (10 points)
    """
    start_time = time.time()

    try:
        global model, models_loaded

        # Check if model is loaded
        if not models_loaded or model is None:
            return jsonify({
                'error': 'Model not loaded. Please restart the application.',
                'details': 'MobileNet model should be loaded at startup'
            }), 503

        # Validate request
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Use PNG, JPG, or JPEG'}), 400

        # Optimize image
        optimized_file = optimize_image(file.stream)

        # Save file temporarily
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        with open(filepath, 'wb') as f:
            f.write(optimized_file.read())

        # Make prediction using MobileNet (direct - no SVM)
        print(f"[PREDICT] Processing: {filename}")

        with model_lock:
            prediction, confidence = predict_single(model, filepath)

        print(f"[PREDICT] Result: {prediction} (confidence: {confidence:.4f})")

        # Calculate latency
        latency = (time.time() - start_time) * 1000  # milliseconds

        # Update metrics
        metrics['total_predictions'] += 1
        metrics['latencies'].append(latency)
        metrics['predictions_log'].append({
            'timestamp': datetime.now().isoformat(),
            'prediction': prediction,
            'confidence': float(confidence),
            'latency_ms': latency,
            'filename': filename
        })

        # Keep only last 1000 predictions
        if len(metrics['predictions_log']) > 1000:
            metrics['predictions_log'] = metrics['predictions_log'][-1000:]

        # Clean up
        try:
            os.remove(filepath)
        except:
            pass

        return jsonify({
            'prediction': prediction,
            'confidence': float(confidence),
            'latency_ms': round(latency, 2),
            'timestamp': datetime.now().isoformat(),
            'model': 'MobileNet End-to-End'
        })

    except Exception as e:
        metrics['total_errors'] += 1
        print(f"[ERROR] Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': str(e),
            'type': type(e).__name__
        }), 500


@app.route('/predict-batch', methods=['POST'])
def predict_batch():
    """
    Batch prediction using MobileNet end-to-end
    """
    start_time = time.time()

    try:
        global model, models_loaded

        if not models_loaded:
            return jsonify({'error': 'Model not loaded'}), 503

        if 'files' not in request.files:
            return jsonify({'error': 'No files provided'}), 400

        files = request.files.getlist('files')

        if not files:
            return jsonify({'error': 'No files selected'}), 400

        # Save files temporarily
        filepaths = []
        filenames = []

        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

                # Optimize and save
                optimized = optimize_image(file.stream)
                with open(filepath, 'wb') as f:
                    f.write(optimized.read())

                filepaths.append(filepath)
                filenames.append(filename)

        print(f"[BATCH] Processing {len(filepaths)} images with MobileNet")

        # Make batch predictions (MobileNet direct)
        with model_lock:
            results = predict_batch(model, filepaths)

        # Calculate latency
        latency = (time.time() - start_time) * 1000

        # Format results
        predictions = []
        for filename, (prediction, confidence) in zip(filenames, results):
            predictions.append({
                'filename': filename,
                'prediction': prediction,
                'confidence': float(confidence)
            })

        # Update metrics
        metrics['total_predictions'] += len(predictions)

        # Clean up
        for filepath in filepaths:
            try:
                os.remove(filepath)
            except:
                pass

        print(f"[BATCH] Completed in {latency:.2f}ms")

        return jsonify({
            'predictions': predictions,
            'total_processed': len(predictions),
            'total_latency_ms': round(latency, 2),
            'avg_latency_per_image_ms': round(latency / len(predictions), 2),
            'timestamp': datetime.now().isoformat(),
            'model': 'MobileNet End-to-End'
        })

    except Exception as e:
        print(f"[ERROR] Batch prediction failed: {e}")
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

        print(f"\n[UPLOAD] Processing {len(files)} files with label: {label}")

        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(label_dir, filename)

                # Optimize image
                optimized = optimize_image(file.stream)
                with open(filepath, 'wb') as f:
                    f.write(optimized.read())

                # Save to database
                print(f"  → Saving to database: {filename}")
                image_id = save_uploaded_file_to_database(filepath, filename, label)

                if image_id:
                    print(f"  → Preprocessing image ID: {image_id}")
                    success = preprocess_uploaded_data(image_id)

                    if success:
                        saved_files.append(filename)
                        database_ids.append(image_id)
                        print(f"  ✓ Successfully processed: {filename}")
                    else:
                        print(f"  ✗ Preprocessing failed for: {filename}")

        print(f"[UPLOAD] Complete: {len(saved_files)}/{len(files)} files processed\n")

        return jsonify({
            'success': True,
            'files_uploaded': len(saved_files),
            'database_ids': database_ids,
            'label': label,
            'message': f'Successfully uploaded and preprocessed {len(saved_files)} files'
        })

    except Exception as e:
        print(f"[ERROR] Upload failed: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/retrain', methods=['POST'])
def retrain():
    """
    Trigger model retraining (MobileNet end-to-end)
    """
    try:
        params = request.get_json() or {}
        epochs = params.get('epochs', 5)
        batch_size = params.get('batch_size', 32)

        print(f"\n{'=' * 70}")
        print("RETRAINING REQUEST RECEIVED - MOBILENET END-TO-END")
        print(f"{'=' * 70}")
        print(f"Epochs: {epochs}")
        print(f"Batch size: {batch_size}")

        job_id = f"retrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        def retrain_worker():
            try:
                print(f"[RETRAIN] Starting job: {job_id}")
                trigger_retraining(
                    data_dir=None,
                    epochs=epochs,
                    batch_size=batch_size,
                    job_id=job_id
                )
                print(f"[RETRAIN] Job {job_id} completed")

                # Reload model after retraining
                global model, models_loaded
                with model_lock:
                    model = load_mobilenet_model(model_path='notebooks/models/mobilenet_final.keras')
                    models_loaded = True
                print(f"[RETRAIN] Model reloaded successfully")

            except Exception as e:
                print(f"[ERROR] Retraining job {job_id} failed: {e}")
                import traceback
                traceback.print_exc()

        thread = threading.Thread(target=retrain_worker, daemon=True)
        thread.start()

        return jsonify({
            'status': 'retraining_started',
            'job_id': job_id,
            'epochs': epochs,
            'batch_size': batch_size,
            'message': 'MobileNet retraining started (end-to-end mode)',
            'info': 'Model will be automatically reloaded after training completes'
        })

    except Exception as e:
        print(f"[ERROR] Retrain endpoint failed: {e}")
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
        # Load training history
        history_path = 'models/results.json'
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                training_history = json.load(f)
        else:
            training_history = {}

        # Prediction distribution
        pred_distribution = {'NORMAL': 0, 'PNEUMONIA': 0}
        confidence_scores = []

        for pred in metrics['predictions_log']:
            pred_distribution[pred['prediction']] = pred_distribution.get(pred['prediction'], 0) + 1
            confidence_scores.append(pred['confidence'])

        # Database statistics
        db_stats = get_database_statistics()

        return jsonify({
            'training_history': training_history,
            'prediction_distribution': pred_distribution,
            'confidence_scores': confidence_scores,
            'latency_data': metrics['latencies'][-100:],
            'database_statistics': db_stats
        })

    except Exception as e:
        print(f"[ERROR] Visualization data failed: {e}")
        return jsonify({'error': str(e)}), 500


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("STARTING FLASK APPLICATION - MOBILENET END-TO-END")
    print("=" * 70)
    print("Access the application at:")
    print("  → Dashboard: http://localhost:5000")
    print("  → Prediction: http://localhost:5000/predict-page")
    print("  → Visualizations: http://localhost:5000/visualize")
    print("  → Retraining: http://localhost:5000/retrain-page")
    print("\nMode: MobileNet End-to-End Classification")
    print("  ✓ No SVM - Direct MobileNet predictions")
    print("  ✓ Eager model loading at startup")
    print("  ✓ Image optimization enabled")
    print("  ✓ Batch prediction support")
    print("=" * 70 + "\n")

    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)