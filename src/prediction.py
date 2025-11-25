"""
prediction.py - Pneumonia Detection Prediction Script (HOG + Random Forest)

Usage:
    python prediction.py --image path/to/image.jpg
    python prediction.py --image path/to/image.jpg --model custom_model.pkl
    python prediction.py --image folder/ --batch  # Predict all images in folder
"""

import os
import sys
import cv2
import numpy as np
import argparse
from skimage.feature import hog
from joblib import load
from pathlib import Path


class PneumoniaPredictor:
    """Pneumonia detection predictor using HOG features and Random Forest"""

    def __init__(self, model_path="../notebooks/random_forest_hog.pkl"):
        """
        Initialize predictor with trained model

        Args:
            model_path: Path to the saved model file
        """
        # Try multiple model paths
        possible_paths = [
            model_path,
            "random_forest_hog.pkl",
            "models/random_forest_base.pkl"
        ]

        loaded = False
        for path in possible_paths:
            if os.path.exists(path):
                print(f"Loading model from {path}...")
                self.model = load(path)
                self.model_path = path
                loaded = True
                break

        if not loaded:
            raise FileNotFoundError(
                f"Model file not found. Tried: {', '.join(possible_paths)}\n"
                f"Please train a model first or specify correct path with --model"
            )

        self.class_names = ['NORMAL', 'PNEUMONIA']
        self.img_size = 128
        print("Model loaded successfully!")
        print(f"  Model type: Random Forest")
        print(f"  Number of estimators: {self.model.n_estimators}")
        print(f"  Classes: {self.class_names}")

    def extract_hog_features(self, img_path):
        """
        Extract HOG features from an image

        Args:
            img_path: Path to the image file

        Returns:
            HOG feature vector
        """
        img = cv2.imread(img_path)

        if img is None:
            raise FileNotFoundError(f"Could not load image: {img_path}")

        # Resize to standard size
        img = cv2.resize(img, (self.img_size, self.img_size))

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Extract HOG features with optimized parameters
        features = hog(
            gray,
            orientations=12,
            pixels_per_cell=(6, 6),
            cells_per_block=(3, 3),
            block_norm="L2-Hys",
            transform_sqrt=True
        )

        return features

    def predict_single(self, image_path, show_confidence=True):
        """
        Predict pneumonia for a single image

        Args:
            image_path: Path to the image
            show_confidence: Whether to show confidence scores

        Returns:
            Dictionary with prediction results
        """
        try:
            # Extract features
            features = self.extract_hog_features(image_path)

            # Make prediction
            prediction = self.model.predict([features])[0]
            probabilities = self.model.predict_proba([features])[0]

            result = {
                'image': os.path.basename(image_path),
                'prediction': self.class_names[prediction],
                'confidence': probabilities[prediction],
                'probabilities': {
                    self.class_names[i]: prob
                    for i, prob in enumerate(probabilities)
                }
            }

            return result

        except Exception as e:
            return {
                'image': os.path.basename(image_path),
                'error': str(e)
            }

    def predict_batch(self, folder_path):
        """
        Predict pneumonia for all images in a folder

        Args:
            folder_path: Path to folder containing images

        Returns:
            List of prediction results
        """
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        image_files = [
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.lower().endswith(image_extensions)
        ]

        if not image_files:
            print(f"No image files found in {folder_path}")
            return []

        print(f"Found {len(image_files)} images. Processing...")

        results = []
        for img_path in image_files:
            result = self.predict_single(img_path)
            results.append(result)

            # Print result
            if 'error' in result:
                print(f"❌ {result['image']}: ERROR - {result['error']}")
            else:
                emoji = "🔴" if result['prediction'] == 'PNEUMONIA' else "🟢"
                print(f"{emoji} {result['image']}: {result['prediction']} "
                      f"({result['confidence']:.2%} confidence)")

        return results

    def print_detailed_result(self, result):
        """Print detailed prediction result"""
        print("\n" + "=" * 60)
        print(f"Image: {result['image']}")
        print("=" * 60)

        if 'error' in result:
            print(f"❌ ERROR: {result['error']}")
        else:
            # Determine risk level
            if result['prediction'] == 'PNEUMONIA':
                if result['confidence'] > 0.8:
                    risk = "🔴 HIGH RISK"
                elif result['confidence'] > 0.6:
                    risk = "🟠 MODERATE RISK"
                else:
                    risk = "🟡 LOW-MODERATE RISK"
            else:
                risk = "🟢 LOW RISK"

            print(f"\n🔍 Prediction: {result['prediction']}")
            print(f"📊 Confidence: {result['confidence']:.2%}")
            print(f"⚠️  Risk Level: {risk}")

            print(f"\n📈 Class Probabilities:")
            for class_name, prob in result['probabilities'].items():
                bar = "█" * int(prob * 50)
                print(f"  {class_name:12} : {prob:.2%} {bar}")

            # Additional info
            print(f"\n💡 Model Information:")
            print(f"  Feature Extraction: HOG (Histogram of Oriented Gradients)")
            print(f"  Classifier: Random Forest")
            print(f"  Image Size: {self.img_size}x{self.img_size}")

        print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Predict pneumonia from chest X-ray images using HOG + Random Forest',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python prediction.py --image sample.jpg
  python prediction.py --image test_images/ --batch
  python prediction.py --image sample.jpg --model custom_model.pkl
  python prediction.py --image test_images/ --batch --output results.csv
        """
    )
    parser.add_argument(
        '--image', '-i',
        type=str,
        required=True,
        help='Path to image file or folder'
    )
    parser.add_argument(
        '--model', '-m',
        type=str,
        default='models/random_forest_hog.pkl',
        help='Path to trained model file (default: models/random_forest_hog.pkl)'
    )
    parser.add_argument(
        '--batch', '-b',
        action='store_true',
        help='Process all images in the specified folder'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Save results to CSV file'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show detailed information about the model and predictions'
    )

    args = parser.parse_args()

    # Initialize predictor
    try:
        predictor = PneumoniaPredictor(args.model)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    # Make predictions
    if args.batch:
        if not os.path.isdir(args.image):
            print(f"Error: {args.image} is not a directory")
            sys.exit(1)

        results = predictor.predict_batch(args.image)

        # Print summary
        if results:
            correct_predictions = sum(1 for r in results if 'error' not in r)
            pneumonia_count = sum(
                1 for r in results
                if 'error' not in r and r['prediction'] == 'PNEUMONIA'
            )
            normal_count = sum(
                1 for r in results
                if 'error' not in r and r['prediction'] == 'NORMAL'
            )

            print(f"\n{'=' * 60}")
            print(f"📊 BATCH PREDICTION SUMMARY")
            print(f"{'=' * 60}")
            print(f"✅ Successfully processed: {correct_predictions}/{len(results)} images")
            print(f"🔴 Pneumonia detected: {pneumonia_count}")
            print(f"🟢 Normal: {normal_count}")

            if correct_predictions > 0:
                avg_confidence = np.mean([
                    r['confidence'] for r in results if 'error' not in r
                ])
                print(f"📈 Average confidence: {avg_confidence:.2%}")
            print(f"{'=' * 60}")

        # Save to CSV if requested
        if args.output and results:
            import csv
            with open(args.output, 'w', newline='') as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=['image', 'prediction', 'confidence', 'normal_prob', 'pneumonia_prob']
                )
                writer.writeheader()
                for r in results:
                    if 'error' not in r:
                        writer.writerow({
                            'image': r['image'],
                            'prediction': r['prediction'],
                            'confidence': f"{r['confidence']:.4f}",
                            'normal_prob': f"{r['probabilities']['NORMAL']:.4f}",
                            'pneumonia_prob': f"{r['probabilities']['PNEUMONIA']:.4f}"
                        })
            print(f"\n💾 Results saved to {args.output}")

    else:
        if not os.path.isfile(args.image):
            print(f"Error: {args.image} is not a file")
            sys.exit(1)

        result = predictor.predict_single(args.image)
        predictor.print_detailed_result(result)


if __name__ == "__main__":
    main()