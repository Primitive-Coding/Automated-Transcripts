import os
import re

# Data
import json
import pandas as pd
import numpy as np

# YouTube
from pytubefix import YouTube

# Audio
from pydub import AudioSegment
import librosa
from scipy.io import wavfile

# Laughter
from Model.Periphery.laughter import Laughter

# Plotting
import matplotlib.pyplot as plt

# Clip editing
import logging
from moviepy.editor import VideoFileClip, AudioFileClip, TextClip, CompositeVideoClip
from moviepy.config import change_settings

from VideoTranscriptions.video_transcriber import VideoTranscriber


# Optionally, set moviepy logging level
logging.basicConfig(level=logging.ERROR)
change_settings({"FFMPEG_LOG_LEVEL": "error"})
change_settings(
    {"IMAGEMAGICK_BINARY": r"C:\\Program Files\\ImageMagick-7.1.1-Q16-HDRI\\magick.exe"}
)


class YouTubeVideo:
    def __init__(self, url: str, log: bool = False) -> None:
        self.url = url
        self.log = log
        self.yt = YouTube(url)
        self.title = self.yt.title

        # Dataframes
        self.transcripts = pd.DataFrame()
        self.training_data = pd.DataFrame()
        # Paths
        self.audio_path = f"./dataset/youtube/{self.title}/audio"
        self.chunks_path = f"./dataset/youtube/{self.title}/audio-chunks"
        self.clips_path = f"./dataset/youtube/{self.title}/clips"
        self.shorts_path = f"./dataset/youtube/{self.title}/shorts"
        self.transcripts_path = f"./dataset/youtube/{self.title}/transcripts"
        self.video_path = f"./dataset/youtube/{self.title}/video"
        self.training_data_path = f"./training_data/{self.title}"

        self.paths = {
            "audio": self.audio_path,
            "chunks": self.chunks_path,
            "transcripts": self.transcripts_path,
            "video": self.video_path,
            "training_data": self.training_data_path,
        }

        # Create directories
        self._create_directories()
        self._add_to_directory()

    def __str__(self) -> str:
        return f"[{self.title}]  -  [{self.url}]"

    # ===============================
    # Download
    # ===============================
    def download_youtube_video_info(self):

        audio_exists = self._check_if_audio_exists()
        video_exists = self._check_if_video_exists()
        wav_exists = self._check_if_wav_exists()
        transcripts_exist = self._check_if_transcripts_exist()
        chunks_exist = self._check_if_chunks_exist()
        training_data_exists = self._check_if_training_data_exists()
        training_features_exist = self._check_if_training_features_exists()
        if not audio_exists:
            # Download audio for youtube video.
            audio_file = self.download_audio()
            if self.log:
                print("- Downloading Audio...")
        if not video_exists:
            # Download video for youtube video.
            video_file, resolution = self.download_video()
            if self.log:
                print("- Downloading Video...")
        if not wav_exists:
            # Convert mp4 to wav format.
            self._convert_to_wav()
            if self.log:
                print("- Creating Wav Files...")
        if not transcripts_exist:
            self.download_transcripts()
            if self.log:
                print("- Downloading Transcripts...")
        if not chunks_exist:
            self.download_chunks()
            if self.log:
                print("- Creating Audio Chunks...")
        if not training_data_exists:
            self.download_training_data()
            if self.log:
                print("- Creating Training Data...")
        # if not training_features_exist:
        #     self.download_training_features()

    def download_audio(self):
        # Download audio (without video)
        audio_stream = self.yt.streams.filter(
            only_audio=True, file_extension="mp4"
        ).first()
        audio_file = audio_stream.download(
            output_path=self.audio_path,
            filename="audio.mp4",
        )
        audio = AudioSegment.from_file(f"{self.audio_path}/audio.mp4")
        audio.export(f"{self.audio_path}/audio.wav", format="wav")

        return audio_file

    def download_video(self):
        # Download video (without audio)
        video_stream = (
            self.yt.streams.filter(
                adaptive=True, file_extension="mp4", only_video=False
            )
            .order_by("resolution")
            .desc()
            .first()
        )
        video_file = video_stream.download(
            output_path=self.video_path,
            filename="video.mp4",
        )

        return video_file, video_stream.resolution

    # ===============================
    # Training Data
    # ===============================

    def set_training_data(self):
        try:
            df = pd.read_csv(f"{self.training_data_path}/training_data.csv").drop(
                "Unnamed: 0", axis=1
            )
        except FileNotFoundError:
            self.download_training_data()
            self.set_training_data()
        self.training_data = df

    def get_training_data(self):
        if self.training_data.empty:
            self.set_training_data()
        return self.training_data

    def download_training_data(self):
        dirs = os.listdir(self.chunks_path)
        dirs = sorted(dirs, key=self._sort_dir_list)
        transcripts = self.get_transcripts()
        data = {
            "start": [],
            "start_ms": [],
            "end": [],
            "end_ms": [],
            "file": [],
            "volume": [],
            "text": [],
        }
        for i, row in transcripts.iterrows():
            d = dirs[i]
            path = f"{self.chunks_path}/{d}"
            volume = self._get_wav_max_volume(path)
            data["start"].append(row["start"])
            data["start_ms"].append(self._time_to_milliseconds(row["start"]))
            data["end"].append(row["end"])
            data["end_ms"].append(self._time_to_milliseconds(row["end"]))
            data["file"].append(d)
            data["volume"].append(volume)
            data["text"].append(row["text"])
        df = pd.DataFrame(data)
        df.to_csv(f"{self.training_data_path}/training_data.csv")

    # ===============================
    # Transcripts
    # ===============================
    def set_transcripts(self):
        try:
            df = pd.read_csv(f"{self.transcripts_path}/transcripts.csv").drop(
                "Unnamed: 0", axis=1
            )
        except FileNotFoundError:
            self.download_transcripts()
            self.set_transcripts()
        self.transcripts = df

    def get_transcripts(self):
        if self.transcripts.empty:
            self.set_transcripts()
        return self.transcripts

    def download_transcripts(
        self,
    ):

        try:
            captions = self.yt.caption_tracks[0]
        except IndexError:
            captions = self.yt.captions
            print(f"Captions: {captions}\nCaptions not available for '{self.title}'")
            exit()

        captions = captions.generate_srt_captions()
        captions = captions.split("\n")
        cur_index = 0
        cols = ["index", "timestamp", "text"]
        data = {"index": [], "start": [], "end": [], "text": []}
        for c in captions:

            if c == "":
                pass
                cur_index = 0
            else:
                col = cols[cur_index]
                if col == "timestamp":
                    start, end = c.split("-->")
                    start = start.strip(" ").replace(",", ".")
                    end = end.strip(" ").replace(",", ".")
                    start_ms = self._time_to_milliseconds(start)
                    end_ms = self._time_to_milliseconds(end)
                    duration = end_ms - start_ms
                    duration = duration / 2
                    new_end_ms = start_ms + duration
                    end = self._milliseconds_to_time(new_end_ms)

                    data["start"].append(start)
                    data["end"].append(end)

                else:
                    data[col].append(c)
                cur_index += 1

        df = pd.DataFrame(data)
        df.drop("index", axis=1, inplace=True)
        df.to_csv(f"{self.transcripts_path}/transcripts.csv")

    # ===============================
    # Wav Files
    # ===============================
    def _get_wav_max_volume(self, path_to_wav_file: str):
        sample_rate, data = wavfile.read(path_to_wav_file)
        # Check if the audio is stereo or mono
        if len(data.shape) == 2:  # Stereo
            # Calculate the maximum value across both channels
            max_amplitude = np.max(np.abs(data), axis=0).max()
        else:  # Mono
            # Calculate the maximum value in the mono channel
            max_amplitude = np.max(np.abs(data))
        return max_amplitude

    def _convert_to_wav(self):
        sound = AudioSegment.from_file(f"{self.audio_path}/audio.mp4", format="mp4")
        sound.export(f"{self.audio_path}/audio.wav", format="wav")

    # ===============================
    # Audio Features
    # ===============================
    def get_audio_features(self):
        path = f"{self.audio_path}/audio.wav"
        y, sr = librosa.load(path, sr=None)
        # Extract the features
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        # Transpose to have time steps as rows and features as columns.
        mfccs = mfccs.T
        return mfccs

    def encode_transcript(self, text, decode: bool = False):
        """
        Convert characters to numbers.
        """
        vocab_path = "./Model/Periphery/vocab.json"

        with open(vocab_path, "r") as vocab_file:
            vocab_data = json.load(vocab_file)

        # vocab = sorted(set(text))
        if not decode:
            char_to_index = vocab_data["char_to_index"]
            # Convert transcript to numerical format
            # encoded_transcript = np.array([char_to_index[char] for char in text])
            encoded_text = [
                char_to_index[char] for char in text if char in char_to_index
            ]
            return encoded_text
        else:
            index_to_char = vocab_data["index_to_char"]
            # index_to_char = {idx: char for idx, char in enumerate(vocab)}
            # Convert transcript to numerical format
            # Convert back to text from indices
            decoded_text = "".join([index_to_char[str(idx)] for idx in text])
            return decoded_text
        # return encoded_transcript

    def get_vocab(self):

        df = pd.read_csv(f"{self.transcripts_path}/transcripts.csv")
        # Concatenate all rows in the "text" column into a single string
        long_string = " ".join(df["text"])
        vocab = sorted(set(long_string))
        return vocab

    def get_vocab_len(self):
        vocab_path = "./Model/Periphery/vocab.json"
        with open(vocab_path, "r") as vocab_file:
            vocab_data = json.load(vocab_file)

        return len(vocab_data["index_to_char"])

    def get_full_transcript(self, encode: bool = False):
        df = pd.read_csv(f"{self.transcripts_path}/transcripts.csv")
        # Concatenate all rows in the "text" column into a single string
        long_string = " ".join(df["text"])

        if encode:
            long_string = self.encode_transcript(long_string)
        return long_string

    # ===============================
    # Audio Chunks
    # ===============================
    def get_chunk_audio_features(self, file_name: str):
        """
        file_name: str
            Name of file. Expects a wav audio file.
        """
        if "." not in file_name:
            file_name = f"{file_name}.wav"
        path = f"{self.chunks_path}/{file_name}"
        y, sr = librosa.load(path, sr=None)
        # Extract the features
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        # Transpose to have time steps as rows and features as columns.
        return mfccs

    def download_chunks(self):
        """
        Download each segment according to the transcripts.
        Each segment will be saved locally as a '.wav' file.
        """
        # Read wav file
        audio = AudioSegment.from_file(f"{self.audio_path}/audio.wav", format="wav")
        # Get transcripts timestamps.
        transcripts = self.get_transcripts()
        chunk = 1
        for i, row in transcripts.iterrows():
            # Convert timestamps to milliseconds.
            start_ms = self._time_to_milliseconds(row["start"])
            end_ms = self._time_to_milliseconds(row["end"])
            # diff = end_ms - start_ms
            # diff = diff / 2
            # end_ms = start_ms + diff

            # SLice the audio
            audio_segment = audio[start_ms:end_ms]
            # Export the segment.
            output_file = f"chunk{chunk}.wav"
            audio_segment.export(f"{self.chunks_path}/{output_file}", format="wav")
            chunk += 1

    def get_all_chunk_volumes(self, sort_df: bool = False):
        dirs = os.listdir(self.chunks_path)
        dirs = sorted(dirs, key=self._sort_dir_list)
        training_data = self.get_training_data()
        try:
            training_data["volume"]
            overwrite = False
        except KeyError:
            overwrite = True
        if overwrite:
            volumes = []
            for d in dirs:
                path = f"{self.chunks_path}/{d}"
                vol = self._get_wav_max_volume(path)
                volumes.append(vol)
            training_data["volume"] = volumes
            training_data.to_csv(f"{self.training_data_path}/training_data.csv")
        if sort_df:
            df_filtered = training_data[
                ~training_data["text"].isin(["[Music]", "[Applause]"])
            ]
            # Sort the filtered DataFrame based on the 'volume' column
            training_data = df_filtered.sort_values(by="volume", ascending=False)
        return training_data

    def _check_if_chunks_exist(self):
        dirs = os.listdir(self.chunks_path)
        if len(dirs) > 0:
            return True
        else:
            return False

    def get_chunk_file_names(self):
        dirs = os.listdir(self.chunks_path)
        return dirs

    # ===============================
    # Laughter
    # ===============================
    def _filter_laughter_slices(self):
        training_data = self.get_training_data()
        laughs = training_data[training_data["text"] == "[Laughter]"]
        return laughs

    def get_laughter_slices(self, peripheral_frames: int = 5):
        """
        Get clips before and after large moments of laughter.

        peripheral_frames: int
            Number of clips to grab before and after laughter occurs.

        """
        training_data = self.get_training_data()
        laughs = self._filter_laughter_slices()
        laughter_slices = []
        index = 1
        for i, row in laughs.iterrows():
            begin_window = i - peripheral_frames
            end_window = i + peripheral_frames + 1

            p_slice = training_data.iloc[begin_window:end_window]
            path = f"{self.clips_path}/slice{index}.mp4"
            l = Laughter(p_slice, path, self.title)
            laughter_slices.append(l)

            index += 1
        return laughter_slices

    def rank_slices_by_volume(self, laughter_slices: list, return_index: bool = False):
        """
        Loop through each slice, calculate the average volume, and compare to see the loudest average clip.

        laughter_slices: list
            List of 'Laughter' objects.

        returns: Laughter
            Laughter object with the loudest average volume.

        """
        loudest_average = 0
        loudest_index = 0
        index = 0
        averages = []

        # Compute average volumes and identify the loudest slice
        for l_slice in laughter_slices:
            s = l_slice.data
            volume_avg = s["volume"].mean()
            averages.append((index, volume_avg))
            if volume_avg > loudest_average:
                loudest_average = volume_avg
                loudest_index = index
            index += 1

        # Reorder the list based on the computed averages
        # Sort by average volume in descending order
        sorted_indices = [
            index for index, _ in sorted(averages, key=lambda x: x[1], reverse=True)
        ]

        # Create a reordered list of laughter_slices
        reordered_laughter_slices = [laughter_slices[i] for i in sorted_indices]
        # Determine if index is returned, or if list of sorted 'Laughter' objects is returned.
        if return_index:
            return loudest_index
        else:
            return reordered_laughter_slices

    # ===============================
    # Directories
    # ===============================
    def _create_directories(self):
        """
        Create necessary directories for the video.
        """
        os.makedirs(f"./dataset", exist_ok=True)
        os.makedirs(f"./dataset/youtube", exist_ok=True)
        youtube_video_folder = f"./dataset/youtube/{self.title}"
        os.makedirs(youtube_video_folder, exist_ok=True)
        os.makedirs(f"{youtube_video_folder}/audio", exist_ok=True)
        os.makedirs(f"{youtube_video_folder}/audio-chunks", exist_ok=True)
        os.makedirs(f"{youtube_video_folder}/clips", exist_ok=True)
        os.makedirs(f"{youtube_video_folder}/frames", exist_ok=True)
        os.makedirs(f"{youtube_video_folder}/shorts", exist_ok=True)
        os.makedirs(f"{youtube_video_folder}/transcripts", exist_ok=True)
        os.makedirs(f"{youtube_video_folder}/video", exist_ok=True)

        # Create folders for the training data.
        training_video_folder = f"./training_data/{self.title}"
        os.makedirs(training_video_folder, exist_ok=True)

    # ===============================
    # Plotting
    # ===============================

    def plot_laughter_slice(self, l_slice: Laughter):
        """
        Plot volume level of the slice. Marks where the laughter occurs with a red dot.

        l_slice: Laughter
            Object containing laughter data.
        """

        df = l_slice.data
        # Remove milliseconds
        df["end"] = df["end"].str.split(".").str[0]
        # Plotting
        plt.figure(figsize=(10, 6))
        plt.plot(df["end"], df["volume"], marker="o", label="Volume")
        # Highlight rows with '[Laughter]' in the text column
        laughter_df = df[df["text"].str.contains(r"\[Laughter\]", case=False, na=False)]
        plt.plot(laughter_df["end"], laughter_df["volume"], "ro", label="Laughter")
        plt.xlabel("End Time")
        plt.ylabel("Volume")
        plt.xticks(rotation=45)
        plt.title("Volume over Time")
        plt.grid(True)
        plt.legend()
        plt.show()

    def plot_all_laughter_slices(
        self, slices: list, slice_limit: int = -1, highlighted_slice=None
    ):
        """
        Plot all laughter slices volume levels.

        slices: list
            List of 'Laughter' objects.

        returns: None

        """
        # Plotting
        plt.figure(figsize=(12, 6))
        index = 0
        label_index = 1
        for _slice in slices:
            s = _slice.data
            slice_len = len(s)
            plot_idx = [i for i in range(slice_len)]
            if highlighted_slice != None:
                if label_index == highlighted_slice:
                    plt.plot(
                        plot_idx, s["volume"], marker="o", label=f"Slice {label_index}"
                    )
                else:
                    plt.plot(
                        plot_idx,
                        s["volume"],
                        marker="o",
                        linestyle="--",
                        label=f"Slice {label_index}",
                    )
            else:
                plt.plot(
                    plot_idx, s["volume"], marker="o", label=f"Slice {label_index}"
                )
            if slice_limit != -1:
                if index >= slice_limit:
                    break
            label_index += 1
            index += 1
        plt.xlabel("End Time")
        plt.ylabel("Volume")
        plt.xticks(rotation=45)
        plt.title("Volume over Time")
        plt.grid(True)
        plt.legend()
        plt.show()

    # ===============================
    # Utilities
    # ===============================

    def _time_to_seconds(self, t):
        h, m, s = map(float, t.split(":"))
        return h * 3600 + m * 60 + s

    def _time_to_milliseconds(self, t) -> int:
        """
        Function to convert HH:MM:SS.MMM to milliseconds

        t: str
            String of a timestamp.

        returns: int
            Integer representing the timestamp in milliseconds.
        """
        h, m, s = t.split(":")
        s, ms = s.split(".")
        return (int(h) * 3600 + int(m) * 60 + int(s)) * 1000 + int(ms)

    def _milliseconds_to_time(self, ms):
        """
        Function to convert HH:MM:SS.MMM to milliseconds

        ms: int
            Integer representing a timestamp in milliseconds.

        returns: str
            String representing 'ms' as a timestamp.
        """
        # Calculate hours, minutes, and seconds
        seconds, milliseconds = divmod(ms, 1000)
        minutes, seconds = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        # Format the time as HH:MM:SS.sss
        time_str = f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}.{int(milliseconds):03}"
        return time_str

    def _add_to_directory(self):
        path = "./dataset/directory/directory.csv"
        try:
            df = pd.read_csv(path)
            d = df[df["title"] == self.title]
            if d.empty:
                new_df = pd.DataFrame(
                    {
                        "title": [self.title],
                        "url": [self.url],
                        "dataset_name": "youtube",
                    }
                )

                df = pd.concat([df, new_df], ignore_index=True)
                df = df.drop("Unnamed: 0", axis=1)
                df.to_csv(path)
        except FileNotFoundError:
            df = pd.DataFrame(
                {
                    "title": [self.title],
                    "url": [self.url],
                    "dataset_name": "youtube",
                }
            )
            df.to_csv(path)
        except pd.errors.EmptyDataError:
            df = pd.DataFrame(
                {
                    "title": [self.title],
                    "url": [self.url],
                    "dataset_name": "youtube",
                }
            )
            df.to_csv(path)

    def _check_if_video_exists(self) -> bool:
        dirs = os.listdir(self.video_path)
        for d in dirs:
            if d == "video.mp4":
                return True
        return False

    def _check_if_training_data_exists(self) -> bool:
        try:
            df = pd.read_csv(f"{self.training_data_path}/training_data.csv")
            if df.empty:
                return False
            else:
                return True
        except FileNotFoundError:
            return False

    def _check_if_training_features_exists(self) -> bool:
        try:
            df = pd.read_csv(f"{self.training_data_path}/training_features.csv")
            if df.empty:
                return False
            else:
                return True
        except FileNotFoundError:
            return False

    def _check_if_audio_exists(self) -> bool:
        dirs = os.listdir(self.audio_path)
        for d in dirs:
            if d == "audio.mp4":
                return True
        return False

    def _check_if_transcripts_exist(self) -> bool:
        dirs = os.listdir(self.transcripts_path)
        for d in dirs:
            if d == "transcripts.csv":
                return True
        return False

    def _check_if_wav_exists(self) -> bool:
        dirs = os.listdir(self.audio_path)
        for d in dirs:
            if d == "audio.wav":
                return True
        return False

    def _sort_dir_list(self, dirs: list) -> list:
        # This will split the string into a list of strings and integers
        return [
            int(part) if part.isdigit() else part for part in re.split(r"(\d+)", dirs)
        ]

    def _get_paths(self) -> dict:
        return self.paths

    def _check_if_data_downloaded(self) -> bool:

        training_data = self._check_if_training_data_exists()
        audio = self._check_if_audio_exists()
        transcripts = self._check_if_transcripts_exist()
        chunks = self._check_if_chunks_exist()
        wav = self._check_if_wav_exists()

        if training_data and audio and transcripts and chunks and wav:
            return True
        else:
            return False

    # ===============================
    # Demo Clips
    # ===============================
    def export_all_clips(self):

        laughter_slices = self.get_laughter_slices()

        label_index = 1
        for ls in laughter_slices:
            path = f"{self.clips_path}/slice{label_index}.mp4"

            clip_exists = self.check_if_clip_exists(path)

            if not clip_exists:
                try:
                    self.export_clip(ls, path)
                except OSError:
                    pass
            label_index += 1

    def export_clip(
        self, l_slice: Laughter, path_to_export: str, verbose: bool = False
    ):
        """
        Create a demo clip based on a laughter slice.
        """
        df = l_slice.data
        # Convert timestamps to seconds.
        start_time = df["start"].iloc[0]
        end_time = df["end"].iloc[-1]
        clip_start = self._time_to_seconds(start_time)
        clip_end = self._time_to_seconds(end_time)

        # Load video clip.
        video_file = f"{self.video_path}/video.mp4"
        video_clip = VideoFileClip(video_file)
        # Load audio
        audio_file = f"{self.audio_path}/audio.wav"
        audio_clip = AudioFileClip(audio_file)

        clipped_video = video_clip.subclip(clip_start, clip_end)
        clipped_audio = audio_clip.subclip(clip_start, clip_end)

        # Combine video and audio, handling potential issues
        final_clip = clipped_video.set_audio(clipped_audio)

        # final_clip = CompositeVideoClip([final_clip] + text_clips)

        # Save with a different filename to avoid conflicts
        final_clip.write_videofile(
            path_to_export,
            codec="libx264",
            audio_codec="aac",
            verbose=verbose,
            logger=None,
        )

    def convert_to_short(self):

        laughter_slices = self.get_laughter_slices()
        index = 1
        for ls in laughter_slices:

            path = f"{self.shorts_path}/short{index}.mp4"

            short_exists = self.check_if_clip_exists(path)

            if not short_exists:
                vt = VideoTranscriber(ls)
                vt.run(path)
            index += 1

    def check_if_clip_exists(self, path_to_clip):
        if os.path.exists(path_to_clip):
            return True
        else:
            return False

    def check_if_clips_exist(self) -> bool:
        dirs = os.listdir(self.clips_path)
        if len(dirs) > 0:
            return True
        else:
            return False
