import zipfile
import os

zip_path = 'captcha_dataset.zip'

extract_dir = 'samples'

os.makedirs(extract_dir, exist_ok=True)

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)

print(f"Archive '{zip_path}' extracted to '{extract_dir}'.")
