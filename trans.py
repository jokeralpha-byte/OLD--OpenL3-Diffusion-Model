import os
import soundfile as sf
import shutil

def convert_audio_format(input_file, output_file):
    """
    Convert input audio file to WAV format using soundfile.
    """
    try:
        # Read the audio file
        audio, samplerate = sf.read(input_file)
        # Save the audio as WAV
        sf.write(output_file, audio, samplerate)
        print(f"Successfully converted {input_file} to {output_file}")
    except Exception as e:
        print(f"Error converting {input_file}: {e}")
        # Remove the directory containing the input file
        shutil.rmtree(os.path.dirname(input_file))

def convert_directory(directory):
    """
    Traverse all subdirectories of a given directory and convert all MP3 and FLAC files to WAV.
    """
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            # Check if the file is mp3 or flac
            if file.lower().endswith(('.mp3', '.flac')):
                output_path = os.path.splitext(file_path)[0] + '.wav'
                convert_audio_format(file_path, output_path)

if __name__ == "__main__":
    directory_to_search = "C:/Users/Lenovo/Desktop/musicshort"  # Change this to the root folder you want to search
    convert_directory(directory_to_search)