import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

## ALGORITHM TO DETECT PEAKS AND LOWS OF A TAKE

def getStartingEnd(path, object_part, video_length, hammer_mass=2.1, min_prominence=5, plot_active=False, speed_limit=250,
                   gradient_threshold=1000, max_time_to_hit = 400, range_for_gradient = 3):

    # parameters
    g = 9.81  # gravitational acceleration (m/s^2)

    # Load the dataset
    dataset_array = np.load(path, allow_pickle=True)
    hammer_hit_set = []

    # Loop through each dataset
    for data_index, data in enumerate(dataset_array):
        if data_index >= 0:
            dataset_name = data["dataset_name"]
            z_coord = data[object_part][1]
            total_duration_ms = video_length[data_index]

            # Ensure `z_coord` is a numeric array
            try:
                z_coord = np.array(z_coord, dtype=float)
            except ValueError:
                z_coord = np.array([float(value) if str(value).strip() else np.nan for value in z_coord])
                z_coord = z_coord[~np.isnan(z_coord)]  # Remove NaN values

            # Detect peaks (high points) and valleys (low points)
            peaks, _ = find_peaks(z_coord, prominence=min_prominence)
            low_points, _ = find_peaks(-z_coord, prominence=min_prominence)

            # Calculate time per data point
            time_per_point = total_duration_ms / len(z_coord)

            # Filter to only include transitions from peaks to valleys
            hammer_hits = []
            # Scale factor for the midpoint of the hammer
            height_scaling_factor = 2  # Coordinate is in the middle of the hammer length

            # Loop to detect hits and adjust height
            for peak in peaks:
                subsequent_lows = low_points[low_points > peak]
                if len(subsequent_lows) > 0:
                    low = subsequent_lows[0]  # Take the first low after the peak

                    # Calculate height difference and adjust for midpoint
                    delta_h_measured = z_coord[peak] - z_coord[low]  # Measured height difference in mm
                    delta_h_actual = delta_h_measured * height_scaling_factor  # Adjusted height

                    delta_t = (low - peak) * time_per_point / 1000  # Time difference in seconds

                    # Calculate speed before the low
                    speed_before_low = delta_h_actual / delta_t  # Speed in mm/s

                    # Calculate acceleration at peak (before impact)
                    acceleration = speed_before_low / delta_t  # Acceleration in mm/s^2

                    # Calculate force just before the impact (at peak) (F = m * a)
                    force_before_impact = hammer_mass * acceleration / 1000  # Force in Newtons

                    # Analyze the gradient after the low point
                    z_after_low = z_coord[low:low + range_for_gradient]
                    if len(z_after_low) >= 2:
                        gradient_after_low = np.gradient(z_after_low) / (time_per_point / 1000)  # Gradient in mm/s

                        # Check for sharp spike
                        sharp_spike = np.max(gradient_after_low) > gradient_threshold
                    else:
                        sharp_spike = False

                    # calculate the time between start and end
                    start_end_diff = peak*time_per_point - low*time_per_point

                    # Classify as hammer hit based on speed threshold and sharp spike
                    if speed_before_low > speed_limit and sharp_spike and start_end_diff < max_time_to_hit:
                        hammer_hits.append((peak, low, time_per_point, speed_before_low, force_before_impact, delta_h_actual, start_end_diff))

            if plot_active:

                # Plot the data with adjusted x-axis
                time_array = np.arange(len(z_coord)) * time_per_point
                plt.figure(figsize=(10, 6))
                plt.plot(time_array, z_coord, label='Z-Coordinate')

                # Plot all peaks and lows
                plt.scatter(time_array[peaks], z_coord[peaks], color='orange', label='All Peaks', zorder=5)
                plt.scatter(time_array[low_points], z_coord[low_points], color='purple', label='All Lows', zorder=5)

                # Highlight hammer hits with double circle
                for hit in hammer_hits:
                    peak, low, time_per_point, speed_before_low, force_before_impact, elta_h_actual, start_end_diff = hit
                    plt.scatter(time_array[peak], z_coord[peak], facecolors='none', edgecolors='red', s=200, label='Hammer Hit (Peak)' if hit == hammer_hits[0] else "", zorder=6)
                    plt.scatter(time_array[low], z_coord[low], color='blue', label='Hit Low' if hit == hammer_hits[0] else "", zorder=6)

                plt.title(f'Hammer Swing Data of {dataset_name}')
                plt.xlabel('Time (ms)')
                plt.ylabel('Y-Coordinate')
                plt.legend()
                plt.grid()
                plt.show()

            hammer_hit_set.append((hammer_hits, dataset_name))

    return hammer_hit_set

if __name__ == "__main__":

    # Prepare the real data
    realHammerHits = {
        # Add all your realHammerHits datasets here
        '20 mm_20 2024-11-11 01.03.44 PM_005.csv': [(26, 46), (61, 80), (97, 110), (168, 186), (230, 256)],
        '20 mm_20 2024-11-11 01.03.44 PM_006.csv': [(6, 25), (42, 53), (72, 83), (179, 197), (227, 242)],
        # Add remaining datasets...
    }

    parameters = {
        'min_prominence': 0,
        'speed_limit': 60,
        'gradient_threshold': 26,
        'max_time_to_hit': 300,
        'gradient_range': 3
    }

    # Prepare the data to be loaded
    video_length_20mm = [3360, 2980, 3460, 3280, 3800]
    video_length_40mm = [4820, 5370, 4470, 5030, 3930]
    video_length_80mm = [4390, 6510, 5200, 4470, 5980]
    video_length_set = [video_length_20mm, video_length_40mm, video_length_80mm]

    dataset_20 = "20mm_preprocessed.npy"
    dataset_40 = "40mm_preprocessed.npy"
    dataset_80 = "80mm_preprocessed.npy"
    datasets = [dataset_20, dataset_40, dataset_80]

    # To store guessed hits
    guessedHammerHits = {}

    # Process datasets with the given parameters
    for pos, dataset in enumerate(datasets):
        hits_dataset = getStartingEnd(
            dataset,
            object_part = "Hammer",
            video_length = video_length_set[pos],
            hammer_mass=2.1,
            min_prominence=parameters["min_prominence"],
            plot_active=True,
            speed_limit=parameters["speed_limit"],
            gradient_threshold=parameters["gradient_threshold"],
            max_time_to_hit=parameters["max_time_to_hit"],
            range_for_gradient=parameters["gradient_range"]
        )
        for hit_takes in hits_dataset:
            dataset_name = hit_takes[-1]
            hammer_hits = hit_takes[0]
            guessedHammerHits[dataset_name] = [(peak, low) for peak, low, _, _, _ in hammer_hits]

    # Compare real and guessed hits
    for dataset_name, real_hits in realHammerHits.items():
        guessed_hits = guessedHammerHits.get(dataset_name, [])

        real_set = set(real_hits)
        guessed_set = set(guessed_hits)

        true_positives = real_set & guessed_set
        false_positives = guessed_set - real_set
        false_negatives = real_set - guessed_set

        precision = len(true_positives) / (len(true_positives) + len(false_positives)) if true_positives else 0
        recall = len(true_positives) / (len(true_positives) + len(false_negatives)) if true_positives else 0
        f1_score = (2 * precision * recall) / (precision + recall) if precision + recall > 0 else 0

        # Print results for the dataset
        print(f"Dataset: {dataset_name}")
        print(f"  F1 Score: {f1_score:.2f}")
        print(f"  True Positives (Correct): {sorted(true_positives)}")
        print(f"  False Positives (Wrongly Added): {sorted(false_positives)}")
        print(f"  False Negatives (Missed): {sorted(false_negatives)}")
        print("-" * 40)

    print("Overall Evaluation Complete!")
