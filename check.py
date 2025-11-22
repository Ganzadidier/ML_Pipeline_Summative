from PIL import Image
import os

root = "data"  # <-- change to your dataset root folder

bad_files = []

for subdir, dirs, files in os.walk(root):
    for file in files:
        filepath = os.path.join(subdir, file)
        try:
            img = Image.open(filepath)
            img.verify()
        except Exception:
            bad_files.append(filepath)

print("Invalid image files:")
for f in bad_files:
    print(f)
