import numpy as np
from GetDownTime import getStartingEnd
from tqdm import tqdm
from sklearn.metrics import f1_score

## USE OF ALGORITHM TO DETECT PEAKS AND LOWS OF A TAKE

# the real data
realHammerHits = {'20 mm_20 2024-11-11 01.03.44 PM_005.csv': [(26, 46), (61, 80), (97, 110), (168, 186), (230, 256)],
'20 mm_20 2024-11-11 01.03.44 PM_006.csv': [(6, 25), (42, 53), (72, 83), (179, 197), (227, 242)],
'20 mm_20 2024-11-11 01.03.44 PM_007.csv': [(47, 61), (74, 87), (153, 171), (196, 213), (236, 256)],
'20 mm_20 2024-11-11 01.03.44 PM_008.csv': [(14, 41), (53, 64), (75, 86), (154, 172), (196, 214), (234, 250)],
'20 mm_20 2024-11-11 01.03.44 PM_009.csv': [(7, 22), (35, 48), (62, 75), (91, 104), (199, 215), (238, 256), (278, 299), (321, 337)],
'40 mm_40 2024-11-11 01.03.44 PM_010.csv': [(46, 57), (75, 87), (107, 122), (231, 248), (277, 305), (331, 351), (379, 399), (428, 449)],
'40 mm_40 2024-11-11 01.03.44 PM_011.csv': [(13, 35), (46, 58), (70, 80), (192, 207), (235, 254), (285, 311), (335, 353), (381, 401), (426, 447), (470, 485), (511, 529)],
'40 mm_40 2024-11-11 01.03.44 PM_012.csv': [(12, 29), (39, 54), (118, 136), (162, 178), (200, 216), (236, 251), (273, 289), (334, 347), (377, 403)],
'40 mm_40 2024-11-11 01.03.44 PM_013.csv': [(29, 40), (95, 110), (128, 142), (193, 208), (230, 243), (263, 278), (296, 309), (327, 340), (358, 372), (403, 427), (455, 474)],
'40 mm_40 2024-11-11 01.03.44 PM_014.csv': [(6, 23), (40, 56), (110, 128), (151, 169), (190, 206), (225, 238), (258, 273), (307, 330)],
'80 mm_80 2024-11-11 01.03.44 PM_001.csv': [(89, 115), (151, 169), (203, 225), (282, 303)],
'80 mm_80 2024-11-11 01.03.44 PM_002.csv': [(55, 75), (98, 116), (205, 223), (244, 259), (279, 292), (386, 404), (433, 451), (483, 503), (532, 552), (583, 602)],
'80 mm_80 2024-11-11 01.03.44 PM_003.csv': [(22, 39), (57, 73), (94, 109), (175, 193), (217, 234), (308, 326), (362, 382), (414, 435), (468, 487)],
'80 mm_80 2024-11-11 01.03.44 PM_004.csv': [(28, 49), (118, 134), (165, 182), (208, 226), (253, 273), (312, 330)],
'80 mm_80 2024-11-11 01.03.44 PM_011.csv': [(9, 25), (40, 52), (70, 82), (99, 110), (128, 140), (158, 169), (248, 262), (329, 347), (391, 411), (448, 467), (501, 520), (553, 570)]}

# prepare the data to be loaded
video_length_20mm = [3360, 2980, 3460, 3280, 3800]
video_length_40mm = [4820, 5370, 4470, 5030, 3930]
video_length_80mm = [4390, 6510, 5200, 4470, 5980]
video_length_set = [video_length_20mm, video_length_40mm, video_length_80mm]

dataset_20 = "20mm_preprocessed.npy"
dataset_40 = "40mm_preprocessed.npy"
dataset_80 = "80mm_preprocessed.npy"
datasets = [dataset_20, dataset_40, dataset_80]

# Define the search space for parameters
search_space = {
    "min_prominence": [0, 1, 2, 3, 4, 5],
    "speed_limit": list(range(60, 100, 3)),
    "gradient_threshold": list(range(20, 50, 3)),
    "max_time_to_hit": [300, 400, 500, 600],
    "gradient_range" : [1, 2, 3, 4, 5, 6, 7, 8]
}

# Calculate the total number of combinations
total_combinations = (
    len(search_space["min_prominence"]) *
    len(search_space["speed_limit"]) *
    len(search_space["gradient_threshold"]) *
    len(search_space["max_time_to_hit"]) *
    len(search_space["gradient_range"])
)

# Function to evaluate the guessed hits against the real hits
def evaluate_guessed_hits(real_hits, guessed_hits):
    """Compares real and guessed hits using F1-score as a metric."""
    dataset_scores = []
    for dataset_name, real_hit_list in real_hits.items():
        guessed_hit_list = guessed_hits.get(dataset_name, [])

        # Convert to sets for comparison
        real_set = set(real_hit_list)
        guessed_set = set(guessed_hit_list)

        # Calculate precision, recall, and F1-score
        true_positives = len(real_set & guessed_set)
        false_positives = len(guessed_set - real_set)
        false_negatives = len(real_set - guessed_set)

        precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0
        f1 = (2 * precision * recall) / (precision + recall) if precision + recall > 0 else 0

        dataset_scores.append(f1)
    return np.mean(dataset_scores)  # Return average F1-score across datasets


# Initialize the progress bar
with tqdm(total=total_combinations, desc="Parameter Search Progress") as pbar:
    best_params = None
    best_score = -1

    for min_prominence in search_space["min_prominence"]:
        for speed_limit in search_space["speed_limit"]:
            for gradient_threshold in search_space["gradient_threshold"]:
                for max_time_to_hit in search_space["max_time_to_hit"]:
                    for gradient_range in search_space["gradient_range"]:
                        # Test current parameters
                        guessedHammerHits = {}
                        for pos, dataset in enumerate(datasets):
                            hits_dataset = getStartingEnd(
                                dataset,
                                video_length_set[pos],
                                hammer_mass=2.1,
                                min_prominence=min_prominence,
                                plot_active=False,
                                speed_limit=speed_limit,
                                gradient_threshold=gradient_threshold,
                                max_time_to_hit=max_time_to_hit,
                                range_for_gradient=gradient_range
                            )
                            for hit_takes in hits_dataset:
                                dataset_name = hit_takes[-1]
                                hammer_hits = hit_takes[0]
                                current_take = [(peak, low) for peak, low, _, _, _ in hammer_hits]
                                guessedHammerHits[dataset_name] = current_take

                        # Evaluate guessed hits
                        score = evaluate_guessed_hits(realHammerHits, guessedHammerHits)

                        # Update the best parameters if the score improves
                        if score > best_score:
                            best_score = score
                            best_params = {
                                "min_prominence": min_prominence,
                                "speed_limit": speed_limit,
                                "gradient_threshold": gradient_threshold,
                                "max_time_to_hit": max_time_to_hit,
                                "gradient_range": gradient_range,
                            }

                        # Update progress bar
                        pbar.update(1)
# Output the best parameters
print("Best Parameters Found:")
print(best_params)
print(f"Best F1 Score: {best_score:.4f}")
