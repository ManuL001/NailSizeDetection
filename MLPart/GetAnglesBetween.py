import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

## ALGORITHM TO DETECT ANGLES AT PEAKS AND LOWS OF A TAKE

def find_closest_extreme_points(xy_points, extreme_points):

    closest_points = []

    for point in xy_points:
        distances = [np.linalg.norm(np.array(point) - np.array(extreme)) for extreme in extreme_points]
        closest_point = extreme_points[np.argmin(distances)]
        closest_points.append(closest_point)

    return closest_points


def find_next_biggest_point(data, reference_point):

    # Filter points greater than the reference point
    candidates = [(idx, value) for idx, value in enumerate(data) if value > reference_point]

    # Find the smallest of the candidates (closest to the reference point)
    if candidates:
        next_biggest = min(candidates, key=lambda x: x[1])
        return next_biggest[1]  # Return as (index, value)
    else:
        return None  # No point larger than the reference point


def find_nearest_point(data, reference_point):

    # Validate inputs
    if not isinstance(data, (list, np.ndarray)):
        raise TypeError(f"Expected 'data' to be a list or numpy array, got {type(data)}")
    if not isinstance(reference_point, (int, float, np.integer, np.floating)):
        raise TypeError(f"Expected 'reference_point' to be a number, got {type(reference_point)}")

    # Calculate the nearest point
    nearest_index = min(range(len(data)), key=lambda idx: abs(data[idx] - reference_point))
    return data[nearest_index]


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
    # Load the dataset
    dataset_array = np.load(path, allow_pickle=True)
    angles = []

    for data_index, data in enumerate(dataset_array):
        # Extract the dataset name
        dataset_name = data["dataset_name"]

        # Clean and convert data to float, replacing empty strings with NaN
        def clean_data(entry):
            cleaned_entry = np.array(entry, dtype=object)
            cleaned_entry[cleaned_entry == ''] = np.nan  # Replace empty strings with NaN
            return np.array(cleaned_entry, dtype=float)  # Convert to float

        shoulder = clean_data(data["Skeleton_25_marker:RShoulder"])
        r_upper_arm = clean_data(data["Skeleton_25_marker:RFArm"])
        r_fore_arm = clean_data(data["Skeleton_25_marker:RHand"])

        # Transpose to get a list of (x, y, z) tuples for each timepoint
        shoulder = list(zip(shoulder[0], shoulder[1], shoulder[2]))
        r_upper_arm = list(zip(r_upper_arm[0], r_upper_arm[1], r_upper_arm[2]))
        r_fore_arm = list(zip(r_fore_arm[0], r_fore_arm[1], r_fore_arm[2]))

        # Compute angles at each timepoint
        angles_for_this_dataset = []
        for t in range(len(shoulder)):
            # Extract coordinate triplets
            S = shoulder[t]
            U = r_upper_arm[t]
            F = r_fore_arm[t]

            # Angle at 'U' formed by S-U-F
            elbow_angle_deg = angle_at_point(S, U, F)
            angles_for_this_dataset.append(elbow_angle_deg)

        angles.append({
            "dataset_name": dataset_name,
            "elbow_angles": angles_for_this_dataset,
            "elbow_angles_hammer_start": [],
            "elbow_angles_nail_hit": [],
            "elbow_angles_hammer_start_widest": [],
            "elbow_angles_nail_hit_widest": []
        })

        # Detect peaks (high points) and valleys (low points)
        peaks, _ = find_peaks(angles_for_this_dataset, prominence=0)
        low_points, _ = find_peaks([-1*angle for angle in angles_for_this_dataset], prominence=0)

        # Compute timing for the x-axis
        total_duration_ms = video_length_set[data_index]
        time_per_point = total_duration_ms / len(shoulder)

        # Generate time array for plotting
        time_array = np.arange(len(shoulder)) * time_per_point

        # Plot the data
        plt.figure(figsize=(10, 6))
        plt.plot(time_array, angles_for_this_dataset, label='Elbow Angle')

        hit_peak = hits[dataset_name][-2]  # List of peak times in ms
        hit_lows = hits[dataset_name][-1]  # List of low times in ms

        # Track if the label has already been used
        peak_label_used = False
        low_label_used = False

        for index, hit_index in enumerate(hit_peak):

            # get the angles when the hammer is hit
            angles[-1]["elbow_angles_hammer_start"].append(angles_for_this_dataset[hit_index])

            if show_hit:
                # Plot the peak
                plt.scatter(
                    time_array[hit_index],  # Time in ms
                    angles_for_this_dataset[hit_index],  # Corresponding angle
                    facecolors='none',
                    edgecolors='red',
                    s=200,
                    label='Hammer Hit (Peak)' if not peak_label_used else None,
                    zorder=6
                )
                peak_label_used = True  # Set label flag to True after the first use

            # Do the same for hit_lows
            low_index = hit_lows[index]
            angles[-1]["elbow_angles_nail_hit"].append(angles_for_this_dataset[low_index])

            if show_hit:
                plt.scatter(
                    time_array[low_index],  # Time in ms
                    angles_for_this_dataset[low_index],  # Corresponding angle
                    color='blue',
                    label='Hit Low' if not low_label_used else None,
                    zorder=6
                )
                low_label_used = True  # Set label flag to True after the first use

        # Highlight the extrema points
        closest_high_extreme_point = find_closest_extreme_points(hit_lows, peaks)
        angles[-1]["elbow_angles_hammer_start_widest"] =  [angles_for_this_dataset[idx] for idx in closest_high_extreme_point]
        if show_hit:
            plt.scatter(
                [time_array[idx] for idx in closest_high_extreme_point],  # Time values of extrema
                [angles_for_this_dataset[idx] for idx in closest_high_extreme_point],  # Angle values of extrema
                color='purple',  # Distinct color for extrema
                marker='x',  # Marker style for extrema points
                label='Extreme Points - highs',
                zorder=5
            )

        closest_low_extreme_point = find_closest_extreme_points(hit_peak, low_points)
        angles[-1]["elbow_angles_nail_hit_widest"] = [angles_for_this_dataset[idx] for idx in closest_low_extreme_point]

        if show_hit:
            plt.scatter(
                [time_array[idx] for idx in closest_low_extreme_point],  # Time values of extrema
                [angles_for_this_dataset[idx] for idx in closest_low_extreme_point],  # Angle values of extrema
                color='yellow',  # Distinct color for extrema
                marker='x',  # Marker style for extrema points
                label='Extreme Points - low',
                zorder=5
            )

        # Add plot labels and legend
        if show_hit:
            plt.title(f'Elbow Angle Analysis ({dataset_name})')
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
            video_length_set = video_length_set[pos]
        )

