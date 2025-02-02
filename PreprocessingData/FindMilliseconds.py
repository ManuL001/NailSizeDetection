import os
import moviepy

VideoFileClip = moviepy.VideoFileClip

def get_video_info(video_path):
    try:
        # Load the video
        clip = VideoFileClip(video_path)
        # Get duration in seconds and FPS
        duration_seconds = clip.duration
        fps = clip.fps
        # Convert duration to milliseconds
        duration_milliseconds = int(duration_seconds * 1000)
        # Close the clip to release resources
        clip.close()
        return duration_milliseconds, fps
    except Exception as e:
        print(f"Error processing file {video_path}: {e}")
        return None, None

def process_videos_in_folder(folder_path):
    results = {}
    # Supported video extensions
    supported_extensions = (".avi", ".mp4", ".mov", ".mkv", ".wmv", ".flv")

    # Iterate over all files in the folder
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        # Check if the file is a video
        if os.path.isfile(file_path) and file_name.lower().endswith(supported_extensions):
            length_in_ms, fps = get_video_info(file_path)
            if length_in_ms is not None and fps is not None:
                results[file_name] = {
                    "duration_milliseconds": length_in_ms,
                    "fps": fps
                }

    return results

# Example usage
folder_path = r"C:\Users\manue\Documents\UNI\Sem 5\Practical Work\Recordings & Data\New Data\80 mm"  # Replace with the path to your folder containing videos
video_infos = process_videos_in_folder(folder_path)

# Display the results
for video_name, info in video_infos.items():
    print(f"{video_name}: {info['duration_milliseconds']} milliseconds, {info['fps']} FPS")

