"""
Automated Dataset Downloader for ML Pipeline
Cross-platform script to download and organize chest X-ray dataset
"""

import os
import sys
import subprocess
import zipfile
import shutil
from pathlib import Path


# Colors for terminal output
class Colors:
    GREEN = '\033[0;32m'
    RED = '\033[0;31m'
    YELLOW = '\033[1;33m'
    NC = '\033[0m'  # No Color

    @staticmethod
    def is_windows():
        return sys.platform.startswith('win')

    @staticmethod
    def print_success(msg):
        if Colors.is_windows():
            print(f"✓ {msg}")
        else:
            print(f"{Colors.GREEN}✓ {msg}{Colors.NC}")

    @staticmethod
    def print_error(msg):
        if Colors.is_windows():
            print(f"✗ {msg}")
        else:
            print(f"{Colors.RED}✗ {msg}{Colors.NC}")

    @staticmethod
    def print_warning(msg):
        if Colors.is_windows():
            print(f"⚠ {msg}")
        else:
            print(f"{Colors.YELLOW}⚠ {msg}{Colors.NC}")


def check_kaggle():
    """Check if Kaggle CLI is installed"""
    try:
        subprocess.run(['kaggle', '--version'],
                       capture_output=True,
                       check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def install_kaggle():
    """Install Kaggle CLI"""
    print("Installing Kaggle CLI...")
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'kaggle'],
                       check=True)
        Colors.print_success("Kaggle CLI installed")
        return True
    except subprocess.CalledProcessError:
        Colors.print_error("Failed to install Kaggle CLI")
        return False


def check_kaggle_credentials():
    """Check if Kaggle credentials are configured"""
    kaggle_json = Path.home() / '.kaggle' / 'kaggle.json'
    return kaggle_json.exists()


def setup_kaggle_instructions():
    """Print instructions for setting up Kaggle credentials"""
    print("\n" + "=" * 60)
    Colors.print_error("Kaggle credentials not found")
    print("=" * 60)
    print("\nPlease setup Kaggle API credentials:\n")
    print("1. Go to: https://www.kaggle.com/account")
    print("2. Scroll to 'API' section")
    print("3. Click 'Create New API Token'")
    print("4. This downloads 'kaggle.json'\n")

    if sys.platform.startswith('win'):
        kaggle_dir = Path.home() / '.kaggle'
        print(f"5. Create folder: {kaggle_dir}")
        print(f"6. Move kaggle.json to: {kaggle_dir / 'kaggle.json'}\n")
        print("Or run these PowerShell commands:")
        print(f"  mkdir {kaggle_dir}")
        print(f"  mv $HOME\\Downloads\\kaggle.json {kaggle_dir}")
    else:
        print("5. Run these commands:")
        print("  mkdir -p ~/.kaggle")
        print("  mv ~/Downloads/kaggle.json ~/.kaggle/")
        print("  chmod 600 ~/.kaggle/kaggle.json")

    print("\nThen run this script again.")
    print("=" * 60 + "\n")


def create_directories():
    """Create necessary directory structure"""
    directories = [
        'data/raw',
        'data/train/NORMAL',
        'data/train/PNEUMONIA',
        'data/val/NORMAL',
        'data/val/PNEUMONIA',
        'data/test/NORMAL',
        'data/test/PNEUMONIA',
        'models',
        'static/uploads'
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)

    Colors.print_success("Directory structure created")


def download_dataset():
    """Download dataset from Kaggle"""
    dataset = 'paultimothymooney/chest-xray-pneumonia'
    download_path = Path('data/raw')
    zip_path = download_path / 'chest-xray-pneumonia.zip'

    if zip_path.exists():
        Colors.print_warning("Dataset already downloaded")
        return True

    print(f"\nDownloading dataset: {dataset}")
    print("Size: ~2 GB (this may take several minutes)")
    print("Please wait...\n")

    try:
        subprocess.run(
            ['kaggle', 'datasets', 'download', '-d', dataset, '-p', str(download_path)],
            check=True
        )
        Colors.print_success("Download complete")
        return True
    except subprocess.CalledProcessError as e:
        Colors.print_error(f"Download failed: {e}")
        return False


def extract_dataset():
    """Extract downloaded dataset"""
    zip_path = Path('data/raw/chest-xray-pneumonia.zip')
    extract_path = Path('data/raw')
    extracted_dir = extract_path / 'chest_xray'

    if extracted_dir.exists():
        Colors.print_warning("Dataset already extracted")
        return True

    print("\nExtracting dataset...")

    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        Colors.print_success("Extraction complete")
        return True
    except Exception as e:
        Colors.print_error(f"Extraction failed: {e}")
        return False


def copy_images(source_dir, dest_dir):
    """Copy images from source to destination"""
    count = 0
    source_path = Path(source_dir)
    dest_path = Path(dest_dir)

    if not source_path.exists():
        return count

    for ext in ['*.jpeg', '*.jpg', '*.png', '*.JPEG', '*.JPG', '*.PNG']:
        for img_file in source_path.glob(ext):
            shutil.copy2(img_file, dest_path)
            count += 1

    return count


def organize_dataset():
    """Organize dataset into train/val/test structure"""
    print("\nOrganizing dataset...")

    base_source = Path('data/raw/chest_xray')

    if not base_source.exists():
        Colors.print_error("Extracted dataset not found")
        return False

    # Mapping of source to destination
    mappings = [
        ('train/NORMAL', 'data/train/NORMAL'),
        ('train/PNEUMONIA', 'data/train/PNEUMONIA'),
        ('val/NORMAL', 'data/val/NORMAL'),
        ('val/PNEUMONIA', 'data/val/PNEUMONIA'),
        ('test/NORMAL', 'data/test/NORMAL'),
        ('test/PNEUMONIA', 'data/test/PNEUMONIA'),
    ]

    for source_subdir, dest_dir in mappings:
        source = base_source / source_subdir
        count = copy_images(source, dest_dir)
        split, label = source_subdir.split('/')
        print(f"  {split}/{label}: {count} images")

    Colors.print_success("Dataset organized")
    return True


def print_statistics():
    """Print dataset statistics"""
    print("\n" + "=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)

    total_all = 0

    for split in ['train', 'val', 'test']:
        normal_path = Path(f'data/{split}/NORMAL')
        pneumonia_path = Path(f'data/{split}/PNEUMONIA')

        normal_count = len(list(normal_path.glob('*.jpeg'))) + len(list(normal_path.glob('*.jpg')))
        pneumonia_count = len(list(pneumonia_path.glob('*.jpeg'))) + len(list(pneumonia_path.glob('*.jpg')))
        total = normal_count + pneumonia_count
        total_all += total

        print(f"\n{split.upper()} SET:")
        print(f"  NORMAL:    {normal_count:,} images")
        print(f"  PNEUMONIA: {pneumonia_count:,} images")
        print(f"  Total:     {total:,} images")

    print(f"\nTOTAL DATASET: {total_all:,} images")
    print("=" * 60)


def main():
    """Main function"""
    print("\n" + "=" * 60)
    print("ML PIPELINE - DATASET SETUP")
    print("=" * 60)
    print()

    # Step 1: Check Kaggle CLI
    print("Step 1: Checking prerequisites...")
    if not check_kaggle():
        Colors.print_warning("Kaggle CLI not found")
        if not install_kaggle():
            return 1
    Colors.print_success("Kaggle CLI ready")

    # Step 2: Check credentials
    print("\nStep 2: Checking Kaggle credentials...")
    if not check_kaggle_credentials():
        setup_kaggle_instructions()
        return 1
    Colors.print_success("Kaggle credentials found")

    # Step 3: Create directories
    print("\nStep 3: Creating directory structure...")
    create_directories()

    # Step 4: Download dataset
    print("\nStep 4: Downloading dataset...")
    if not download_dataset():
        return 1

    # Step 5: Extract dataset
    print("\nStep 5: Extracting dataset...")
    if not extract_dataset():
        return 1

    # Step 6: Organize dataset
    print("\nStep 6: Organizing dataset...")
    if not organize_dataset():
        return 1

    # Step 7: Print statistics
    print_statistics()

    # Success
    print("\n" + "=" * 60)
    Colors.print_success("DATASET SETUP COMPLETE!")
    print("=" * 60)

    print("\nNext steps:")
    print("1. Review images in data/ folders")
    print("2. Train model:")
    print("   python src/model.py --train")
    print("3. Or use Jupyter notebook:")
    print("   jupyter notebook notebooks/pneumonia_classification.ipynb")
    print()

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        Colors.print_error(f"Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)