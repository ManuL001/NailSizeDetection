import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import pandas as pd


def angle_at_point(A, B, C):
    """
    Returns the angle (in degrees) at point B formed by points A-B-C.
    A, B, C each are (x, y, z).
    """
    AB = np.array(B) - np.array(A)  # vector from B to A
    CB = np.array(B) - np.array(C)  # vector from B to C

    # Dot product
    dot_prod = np.dot(AB, CB)
    # Magnitudes
    mag_AB = np.linalg.norm(AB)
    mag_CB = np.linalg.norm(CB)

    # Avoid numerical issues where dot_prod might be slightly out of [-1,1]
    cos_angle = dot_prod / (mag_AB * mag_CB + 1e-9)
    # Clip to valid range for arccos
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    angle_radians = np.arccos(cos_angle)
    angle_degrees = np.degrees(angle_radians)
    return angle_degrees


def getAngleBetween(path, video_length_set, show_hit = False, hits = None):
    # Correct file path
    csv_file_path = r"C:\Users\manue\Documents\UNI\Sem 5\Practical Work\Recordings & Data\New Data\Test\Mappe2.xlsx"

    # Load the Excel file
    df = pd.read_excel(csv_file_path)

    # Clean the data by filling missing values
    df = df.fillna(0)  # Replace missing values with 0 (you can also use other strategies like .interpolate())
    df = df.apply(pd.to_numeric, errors='coerce')  # Convert all data to numeric, set invalid entries to NaN
    df = df.fillna(0)

    # Convert DataFrame to NumPy for further processing
    data = df.to_numpy()

    # Initialize lists for RShoulder, RUArm, and RFArm
    RShoulder = [[], [], []]
    RUArm = [[], [], []]
    RFArm = [[], [], []]

    # Process the data row by row
    for data_index, row in enumerate(data):
        for index, value in enumerate(row):
            if index < 3:
                i = index
                RShoulder[i].append(value)
            elif index < 6:
                i = index - 3
                RUArm[i].append(value)
            elif index < 9:
                i = index - 6  # Adjusted index to fit the RFArm array
                RFArm[i].append(value)

    # Transpose lists to get tuples of (x, y, z) for each timepoint
    shoulder = list(zip(RShoulder[0], RShoulder[1], RShoulder[2]))
    r_upper_arm = list(zip(RUArm[0], RUArm[1], RUArm[2]))
    r_fore_arm = list(zip(RFArm[0], RFArm[1], RFArm[2]))

    # Example: Calculate angles
    angles = []
    for A, B, C in zip(shoulder, r_upper_arm, r_fore_arm):
        angle = angle_at_point(A, B, C)
        angles.append(angle)

    # Compute timing for the x-axis
    total_duration_ms = 5370
    time_per_point = total_duration_ms / len(shoulder)

    # Generate time array for plotting
    time_array = np.arange(len(shoulder)) * time_per_point

    # Plot the data
    plt.figure(figsize=(10, 6))
    plt.plot(time_array, angles, label='Elbow Angle')

    # Add plot labels and legend
    plt.title(f'Elbow Angle Analysis')
    plt.xlabel('Time (ms)')
    plt.ylabel('Angle (degrees)')
    plt.legend()
    plt.grid()
    plt.show()

    return angles


if __name__ == "__main__":

    # Prepare the data to be loaded
    video_length_20mm = [3360, 2980, 3460, 3280, 3800]
    video_length_40mm = [4820, 5370, 4470, 5030, 3930]
    video_length_80mm = [4390, 6510, 5200, 4470, 5980]
    video_length_set = [video_length_20mm, video_length_40mm, video_length_80mm]

    dataset_20 = "20mm_preprocessed.npy"
    dataset_40 = "40mm_preprocessed.npy"
    dataset_80 = "80mm_preprocessed.npy"
    datasets = [dataset_20, dataset_40, dataset_80]

    # Process datasets with the given parameters
    for pos, dataset in enumerate(datasets):
        hits_dataset = getAngleBetween(
            dataset,
            video_length_set = video_length_set
        )

