# convert_model.py

import tf_keras as keras
import os
import sys

def main():
    saved_model_dir = 'models/152_50_0.0001_128/'  # Path to your SavedModel directory
    h5_model_path = 'models/152_50_0.0001_128.h5'  # Desired path for the .h5 file

    # Verify that the SavedModel directory exists
    if not os.path.isdir(saved_model_dir):
        print(f"Error: SavedModel directory '{saved_model_dir}' does not exist.")
        sys.exit(1)

    # Load the SavedModel without compiling to exclude optimizer states
    try:
        saved_model = keras.models.load_model(saved_model_dir, compile=False)
        print("SavedModel loaded successfully.")
    except Exception as e:
        print(f"Error loading SavedModel: {e}")
        sys.exit(1)

    # Save the model in .h5 format without the optimizer
    try:
        saved_model.save(h5_model_path, include_optimizer=False)
        print(f"Model successfully saved to '{h5_model_path}'.")
    except Exception as e:
        print(f"Error saving .h5 model: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()