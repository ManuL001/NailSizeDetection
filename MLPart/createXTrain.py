from GetDownTime import getStartingEnd
from GetAnglesBetween import getAngleBetween

def create_dataset():
    parameters = {'min_prominence': 0, 'speed_limit': 60, 'gradient_threshold': 26, 'max_time_to_hit': 300, 'gradient_range': 3}

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
            plot_active=False,
            speed_limit=parameters["speed_limit"],
            gradient_threshold=parameters["gradient_threshold"],
            max_time_to_hit=parameters["max_time_to_hit"],
            range_for_gradient=parameters["gradient_range"]
        )

        # in here we have = peak, low, time_per_point, speed_before_low, force_before_impact, delta_h_actual, start_end_diff
        for index, hit_takes in enumerate(hits_dataset):
            time_per_point = hit_takes[0][0][2]
            dataset_name = hit_takes[-1]
            num_hammer_hits = len(hit_takes[0])
            peak_time_points = [sublist[0] * time_per_point for sublist in hit_takes[0]]
            peak = [sublist[0] for sublist in hit_takes[0]]
            low_time_points = [sublist[1]  * time_per_point for sublist in hit_takes[0]]
            low = [sublist[1] for sublist in hit_takes[0]]
            time_it_takes_down = [sublist[6] for sublist in hit_takes[0]]
            distance_it_travels_down = [sublist[5] for sublist in hit_takes[0]]
            speed_before_impact = [sublist[6] for sublist in hit_takes[0]]
            guessedHammerHits[dataset_name] = [num_hammer_hits, peak_time_points, low_time_points,
                                               time_it_takes_down, distance_it_travels_down, speed_before_impact, peak, low]

        # add the elbow angle information to the dataset
        angle_datasets = getAngleBetween(dataset, video_length_set[pos], show_hit=True, hits=guessedHammerHits)
        for index, angle_takes in enumerate(angle_datasets):
            guessedHammerHits[angle_takes['dataset_name']].append(angle_takes["elbow_angles_hammer_start"])
            guessedHammerHits[angle_takes['dataset_name']].append(angle_takes["elbow_angles_nail_hit"])
            guessedHammerHits[angle_takes['dataset_name']].append(angle_takes["elbow_angles_hammer_start_widest"])
            guessedHammerHits[angle_takes['dataset_name']].append(angle_takes["elbow_angles_nail_hit_widest"])

    return guessedHammerHits