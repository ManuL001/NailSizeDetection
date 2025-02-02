from PreProcessing import DataPreProcessing
import numpy as np
import os

def list_csv_files(folder_path):
    """
    List all CSV files in the given folder, including their full paths.

    Parameters:
        folder_path (str): The path to the folder to search for CSV files.

    Returns:
        list: A list of full paths to the CSV files.
    """
    csv_files = []

    # Check if the provided path exists
    if not os.path.exists(folder_path):
        print(f"The folder path '{folder_path}' does not exist.")
        return csv_files

    # Walk through the directory and subdirectories
    for root, _, files in os.walk(folder_path):
        for file in files:
            # Check if the file has a .csv extension
            if file.lower().endswith('.csv'):
                # Append the full path to the csv_files list
                full_path = os.path.join(root, file)
                csv_files.append(full_path)

    return csv_files

# ----------------------------------------------------------------------------------------------------------------------
# Main

# Add dataset all the datasets
path_20_mm = r"C:\Users\manue\Documents\UNI\Sem 5\Practical Work\Recordings & Data\New Data\20 mm"
path_40_mm = r"C:\Users\manue\Documents\UNI\Sem 5\Practical Work\Recordings & Data\New Data\40 mm"
path_80_mm = r"C:\Users\manue\Documents\UNI\Sem 5\Practical Work\Recordings & Data\New Data\80 mm"
name_20 = "20mm_preprocessed"
name_40 = "40mm_preprocessed"
name_80 = "80mm_preprocessed"

path_list = [path_20_mm, path_40_mm, path_80_mm]
name_list = [name_20, name_40, name_80]

for pos, path in enumerate(path_list):
    exoskeleton_data_20mm = DataPreProcessing()
    all_xx_mm_csv = list_csv_files(path)

    for index, path in enumerate(all_xx_mm_csv):
        names = path.split("\\")[-2:]
        name = names[0] + "_" + names[1]
        exoskeleton_data_20mm.add_dataset(name, path)

    # Remove columns containing specific data
    exoskeleton_data_20mm.remove_column_data("position_and_rotation", "Rotation")

    # get the data of the requested columns of every dataset
    selected_data = exoskeleton_data_20mm.get_columns(["Skeleton_25_marker:RShoulder", "Skeleton_25_marker:RFArm",
                                       "Skeleton_25_marker:RUArm", "Skeleton_25_marker:RHand", "Hammer"], exact_name=True)

    # convert the data into an xyz array
    for id, dataset in enumerate(selected_data):
        for key in dataset.keys():
            if(key != 'dataset_name'):
                selected_data[id][key] = np.array(selected_data[id][key])

    # save the data
    np.save(name_list[pos], selected_data)
