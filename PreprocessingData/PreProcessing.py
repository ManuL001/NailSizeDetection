import pandas as pd
import numpy as np

class DataPreProcessing:
    def __init__(self):
        # Dictionary to store datasets with their names as keys
        self.datasets = {}

    def add_dataset(self, name, path):
        """
        Adds a dataset to the collection.
        :param name: Name of the dataset
        :param path: Path to the dataset CSV file
        """
        # Load the CSV without specifying multiple header rows
        df = pd.read_csv(path, header=None, delimiter='\t')

        # Dynamically extract headers and data rows
        column_headers = {}
        headers = ["type", "name", "id", "position_and_rotation", "xyzw"]
        for i in range(1, 6):  # Assuming first 5 rows are headers
            column_name = df.iloc[i].iloc[0].split(',')
            header_key = headers[i - 1]  # Use descriptive header names
            column_headers[header_key] = column_name

        # Extract data starting from row 6 onwards
        data = df.iloc[6:]
        data = data[0].apply(self._split_data)

        # Store columns and data separately in the dataset dictionary
        self.datasets[name] = {
            "name": name,  # Store the dataset name within the dataset
            "columns": column_headers,
            "data": data.tolist()
        }
        print(f"Dataset '{name}' added successfully.")

    def process_dataset(self, name):
        """
        Processes the dataset to extract columns and data.
        :param name: Name of the dataset to process
        :return: A dictionary containing columns and data
        """
        if name not in self.datasets:
            raise ValueError(f"Dataset '{name}' not found.")

        return self.datasets[name]

    @staticmethod
    def _split_data(row):
        """
        Helper function to split a row of data by comma.
        """
        return row.split(",")

    def get_dataset(self, name):
        """
        Returns the raw dataset by name.
        :param name: Name of the dataset
        :return: Dictionary containing columns and data of the dataset
        """
        if name not in self.datasets:
            raise ValueError(f"Dataset '{name}' not found.")
        return self.datasets[name]

    def get_all_data_from_name(self, name, specify_dataset=None, exact_name=False):
        """
        Searches all positions in the 'name' column across datasets and returns the matching data.
        :param name: Name to search for in the 'name' column
        :param specify_dataset: Specify a dataset to search in
        :param exact_name: Boolean to specify if the match should be exact
        :return: A list of matching data from all datasets
        """
        matching_data = []
        filtered_columns = []
        datasetnames = []

        for dataset_name, dataset in self.datasets.items():
            matching_data_cur_data = []
            filtered_columns_cur_data = None
            if specify_dataset is None or dataset_name == specify_dataset:
                columns = dataset["columns"]

                if "name" in columns.keys():  # Assuming name corresponds to 'name' column
                    matching_indexes = self._find_matching_indexes(columns["name"], name, exact_name)

                    for index in matching_indexes:
                        data_at_index = [second[index] for second in dataset["data"]]
                        matching_data_cur_data.append(data_at_index)

                    # Filter columns
                    filtered_columns_cur_data = {key: [value[i] for i in matching_indexes] for key, value in columns.items()}

            matching_data.append(matching_data_cur_data)
            filtered_columns.append(filtered_columns_cur_data)
            datasetnames.append(dataset_name)

        return filtered_columns, matching_data, datasetnames

    def remove_column_data(self, column_name, data_name):
        """
        Removes columns from the dataset based on a given data name.
        :param column_name: The name of the column header to search in.
        :param data_name: The data to be removed.
        """
        for dataset_name, dataset in self.datasets.items():
            columns = dataset["columns"]

            # Find all the positions of the data you want to delete
            if column_name in columns.keys():
                chosen_column = columns[column_name]
                matching_indexes = self._find_matching_indexes(chosen_column, data_name, exact_name=True)

                # Remove all the columns that contain this data
                if matching_indexes:
                    for index in sorted(matching_indexes, reverse=True):
                        for key in columns.keys():
                            del columns[key][index]
                        dataset["data"] = [row[:index] + row[index+1:] for row in dataset["data"]]

            # update the changes
            self.datasets[dataset_name]["columns"] = dataset["columns"]
            self.datasets[dataset_name]["data"] = dataset["data"]

    def get_columns(self, retain_column_names, exact_name=False):
        """
        Retains only the specified columns in the dataset, removing all others.

        :param retain_column_names: A list of column names to keep in the dataset.
        """
        dataset = [{} for _ in range(len(self.datasets))]

        for search_name in retain_column_names:
            # Get all data for a specific name
            columns, matching_data, datasetnames = self.get_all_data_from_name(search_name, exact_name=exact_name)

            # loops through the matching data and orders it the right way - only for xyz data
            for cur_dataset_pos in range(len(columns)):
                dataset[cur_dataset_pos]["dataset_name"] = datasetnames[cur_dataset_pos]  # Add dataset name
                dataset[cur_dataset_pos][search_name] = []
                for i, column in enumerate(columns[cur_dataset_pos].get("name", [])):
                    dataset[cur_dataset_pos][search_name].append(matching_data[cur_dataset_pos][i])

        return dataset

    def print_datasets(self):
        """
        Prints the names of all currently loaded datasets and their data.
        """
        if not self.datasets:
            print("No datasets are currently loaded.")
        else:
            print("Currently loaded datasets and their contents:")
            for name, dataset in self.datasets.items():
                print(f"\nDataset Name: {name}")
                print("Columns:")
                for column_name, column_data in dataset["columns"].items():
                    print(f"  {column_name}: {column_data}")
                print("Data:")
                for row in dataset["data"]:
                    print(f"  {row}")

    @staticmethod
    def _find_matching_indexes(names, search_name, exact_name):
        """
        Finds index positions of matching names.
        :param names: List of names to search in
        :param search_name: The name to search for
        :param exact_name: Boolean to specify if the match should be exact
        :return: List of matching index positions
        """
        matching_indexes = []
        for i, name in enumerate(names):
            if (exact_name and search_name == name) or (not exact_name and search_name in name):
                matching_indexes.append(i)
        return matching_indexes


