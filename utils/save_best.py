import os
import csv

# written by zz3128
def save_best_result(file_path,dictionary):
    
    #check if file path exists
    file_exists = os.path.isfile(file_path) and os.path.getsize(file_path) > 0

    fieldnames = dictionary.keys()

    # Open the file to write the header if it does not exist
    if not file_exists:
        with open(file_path, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()

    # Open the file again in append mode and write the row
    with open(file_path, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writerow(dictionary)
    
    dataset_name = dictionary['dataset_name']
    
    print(f"{dataset_name} best result appended successfully to {file_path}")
    
    return dictionary
