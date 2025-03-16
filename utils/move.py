import os
import shutil

# written by zz3128
def move_files(current_path,to_path):
    # Create destination path if it doesn't exist
    if not os.path.exists(to_path):
        os.makedirs(to_path)
        print(f"Folder '{to_path}' created.")

    # Move all files from current todestination path
    for file_name in os.listdir(current_path):
        source_path = os.path.join(current_path, file_name)
        destination_path = os.path.join(to_path, file_name)

        if os.path.isfile(source_path):
            shutil.move(source_path, destination_path)
            print(f"Moved: {file_name}")

    print("All files have been moved.")
    
    return None
