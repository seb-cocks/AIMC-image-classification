import os
from tqdm import tqdm
import numpy as np
import h5py

def load_algorithm_snr_h5s(root_folder, mod_types):
    """
    Loads .h5 spectrogram files from a specific algorithm's snr_X folder,
    filtered by modulation type (FM, PM, HYBRID).

    Parameters:
    - root_folder (str): Path to the snr_X directory (e.g., .../preprocessed_images/cdae/snr_0)
    - mod_types (list): List of modulation categories to include, e.g., ['FM', 'PM']

    Returns:
    - X: np.ndarray of images
    - y: np.ndarray of labels (modulation names as strings)
    """
    X = []
    y = []

    for mod_type in mod_types:
        mod_path = os.path.join(root_folder, mod_type)
        if not os.path.exists(mod_path):
            print(f"⚠️ Warning: {mod_path} does not exist. Skipping.")
            continue

        print(f"📂 Loading from {mod_type}...")
        files = [f for f in os.listdir(mod_path) if f.endswith(".h5")]

        for file in tqdm(files, desc=f"   {mod_type}", unit="file"):
            mod_name = file[:-3]  # Strip '.h5'
            file_path = os.path.join(mod_path, file)

            try:
                with h5py.File(file_path, "r") as h5f:
                    if mod_name not in h5f:
                        print(f"⚠️ Warning: No top-level group named '{mod_name}' in {file_path}")
                        continue
                    group = h5f[mod_name]
                    for key in group.keys():
                        img = np.array(group[key])
                        X.append(img)
                        y.append(mod_name)
            except Exception as e:
                print(f"❌ Failed to load {file_path}: {e}")

    return np.array(X), np.array(y)
