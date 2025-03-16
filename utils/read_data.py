import os
import pandas as pd

# written by kl3606
def tsv_to_csv(input_folder = "UCRArchive_2018", output_folder = "data"):

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Loop through all subfolders in the input folder
    for subfolder in os.listdir(input_folder):
        subfolder_path = os.path.join(input_folder, subfolder)
    
        # Check if it's a directory
        if os.path.isdir(subfolder_path):
            # Loop through all .tsv files in the subfolder
            for file in os.listdir(subfolder_path):
                if file.endswith('.tsv'):
                    tsv_file_path = os.path.join(subfolder_path, file)
                    csv_file_name = file.replace('.tsv', '.csv')
                    csv_file_path = os.path.join(output_folder, csv_file_name)
                
                    # Read the .tsv file and save as .csv
                    df = pd.read_table(tsv_file_path, sep = '\t', header = None, encoding = 'latin-1')
                    df.interpolate(method = 'linear', inplace = True)  # Optional: Handle missing values
                    df.to_csv(csv_file_path, index = False, header = True, encoding = 'utf-8')
                
                    print(f"Converted: {tsv_file_path} to {csv_file_path}")

    print("All files have been converted and saved.")
    
    return None
