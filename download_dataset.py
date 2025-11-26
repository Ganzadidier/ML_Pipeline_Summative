"""
Automated Dataset Downloader for ML Pipeline
Cross-platform script to download and organize chest X-ray dataset
"""

import os
import sys
import zipfile
import shutil
from pathlib import Path
from urllib.parse import urlparse, parse_qsl, urlencode, urlunparse

import requests


# Share links can rotate slightly (different st= parameters), so keep editable.
DROPBOX_SHARED_LINK = (
    "https://www.dropbox.com/scl/fo/iyugbgf5j5csvge5vn1oo/"
    "AEcvzlZpqYxtONMB_NU2z6E?rlkey=3ndh7ragaabbpkg2ktj6rks41&dl=1"
)
PROCESSED_ZIP_NAME = "clean_chest_xray_dataset.zip"


class Colors:
    """Lightweight cross-platform console colors."""

    GREEN = '\033[0;32m'
    RED = '\033[0;31m'
    YELLOW = '\033[1;33m'
    NC = '\033[0m'

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


def ensure_direct_dropbox_link(url: str) -> str:
    """
    Convert any Dropbox preview link into a direct download endpoint.
    For shared folders ('/scl/fo/...') Dropbox already zips the folder when dl=1,
    so we avoid appending extra path segments that would re-route to HTML.
    """
    parsed = urlparse(url)
    scheme = "https"
    netloc = "www.dropbox.com"
    path = parsed.path.rstrip('/')
    query = dict(parse_qsl(parsed.query))
    query["dl"] = "1"
    query.pop("raw", None)
    return urlunparse((scheme, netloc, path, "", urlencode(query), ""))


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


def download_processed_dataset():
    """Download the processed dataset zip from Dropbox."""
    download_path = Path('data/raw')
    download_path.mkdir(parents=True, exist_ok=True)
    zip_path = download_path / PROCESSED_ZIP_NAME

    if zip_path.exists():
        Colors.print_warning(f"{PROCESSED_ZIP_NAME} already exists, skipping download")
        return zip_path

    direct_url = ensure_direct_dropbox_link(DROPBOX_SHARED_LINK)
    print("\nDownloading processed dataset from Dropbox...")
    print(f"Source: {DROPBOX_SHARED_LINK}")

    try:
        with requests.get(direct_url, stream=True, timeout=120) as response:
            response.raise_for_status()

            content_type = response.headers.get("Content-Type", "")
            if "text/html" in content_type:
                Colors.print_warning("Dropbox returned HTML; attempting fallback download URL.")

            with open(zip_path, 'wb') as out_file:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        out_file.write(chunk)

        Colors.print_success("Download complete")
    except Exception as exc:
        Colors.print_error(f"Download failed: {exc}")
        return None

    if not zipfile.is_zipfile(zip_path):
        Colors.print_error(
            "Downloaded file is not a valid ZIP archive. "
            "Please verify the Dropbox link or replace PROCESSED_ZIP_NAME with a .zip file."
        )
        return None

    return zip_path


def extract_dataset(zip_path: Path):
    """Extract downloaded dataset"""
    extract_path = Path('data/raw')
    marker_path = extract_path / '.clean_dataset_extracted'

    if marker_path.exists():
        Colors.print_warning("Dataset already extracted")
        return marker_path.read_text().strip()

    if zip_path is None or not zip_path.exists():
        Colors.print_error("Zip file not found, cannot extract")
        return None

    print("\nExtracting dataset...")

    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        Colors.print_success("Extraction complete")
    except Exception as e:
        Colors.print_error(f"Extraction failed: {e}")
        return None

    # Try to detect the root folder that contains train/val/test
    extracted_root = detect_dataset_root(extract_path)
    if extracted_root is None:
        Colors.print_warning("Could not automatically determine dataset root; using data/raw")
        extracted_root = extract_path

    marker_path.write_text(str(extracted_root))
    return extracted_root


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


def detect_dataset_root(search_path: Path):
    """
    Inspect extracted contents to find the folder that includes the split subfolders.
    """
    candidates = [p for p in search_path.iterdir() if p.is_dir()]

    for candidate in candidates:
        if (candidate / 'train').exists() and (candidate / 'test').exists():
            return candidate

    # Fallback: look deeper one level
    for candidate in candidates:
        for sub in candidate.rglob('*'):
            if sub.is_dir() and (sub / 'train').exists():
                return sub

    return None


def organize_dataset(base_source: Path):
    """Organize dataset into train/val/test structure"""
    print("\nOrganizing dataset...")

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

    # Step 1: Create directories
    print("Step 1: Creating directory structure...")
    create_directories()

    # Step 2: Download dataset from Dropbox
    print("\nStep 2: Downloading processed dataset...")
    zip_path = download_processed_dataset()
    if zip_path is None:
        return 1

    # Step 3: Extract dataset
    print("\nStep 3: Extracting dataset...")
    dataset_root = extract_dataset(zip_path)
    if dataset_root is None:
        return 1

    # Step 4: Organize dataset into expected folders
    print("\nStep 4: Organizing dataset into train/val/test...")
    if not organize_dataset(Path(dataset_root)):
        return 1

    # Step 5: Print statistics
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