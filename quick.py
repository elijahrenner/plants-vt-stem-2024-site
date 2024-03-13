import os

root_directory = "models/"

files_to_delete = [
    "variables.data-00000-of-00001",
    "variables.index",
    "saved_model.pb",
    "keras_metadata.pb",
]

for root, dirs, files in os.walk(root_directory):
    for file_name in files:
        if file_name in files_to_delete:
            file_path = os.path.join(root, file_name)
            os.remove(file_path)
            print(f"Deleted: {file_path}")
