import os
import shutil

import cv2
from moviepy.editor import ImageSequenceClip, AudioFileClip, VideoFileClip
from tqdm import tqdm

from Model.Periphery.laughter import Laughter


FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.8
FONT_THICKNESS = 2


class VideoTranscriber:
    def __init__(self, l: Laughter) -> None:
        self.laugh_slice = l
        self.data = l.data
        self.convert_timestamps()
        self.transcripts_path = f"./transcripts/{l.video_title}"
        os.makedirs(self.transcripts_path, exist_ok=True)
        self._handle_dirs()

        # Vars for transcribing
        self.video_path = l.path
        self.audio_path = ""
        self.text_array = []
        self.fps = 0
        self.char_width = 0

    def convert_timestamps(self):
        anchor_time = self._time_to_seconds(self.data["start"].iloc[0])
        clip_start = []
        clip_end = []
        for i, row in self.data.iterrows():

            start = self._time_to_seconds(row["start"])
            end = self._time_to_seconds(row["end"])
            start = start - anchor_time
            end = end - anchor_time
            clip_start.append(start)
            clip_end.append(end)

        self.data["clip_start"] = clip_start
        self.data["clip_end"] = clip_end

        print(f"Data: {self.data}")

    def transcribe_video(self):
        text = self.data["text"].iloc[0]
        textsize = cv2.getTextSize(text, FONT, FONT_SCALE, FONT_THICKNESS)[0]
        cap = cv2.VideoCapture(self.video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        asp = 16 / 9
        ret, frame = cap.read()
        width = frame[
            :,
            int(int(width - 1 / asp * height) / 2) : width
            - int((width - 1 / asp * height) / 2),
        ].shape[1]
        width = width - (width * 0.1)
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.char_width = int(textsize[0] / len(text))

        for i, row in tqdm(self.data.iterrows()):
            lines = []
            text = row["text"].upper()
            end = row["clip_end"]
            start = row["clip_start"]
            total_frames = int((end - start) * self.fps)
            start = start * self.fps
            total_chars = len(text)
            words = text.split(" ")
            i = 0

            while i < len(words):
                words[i] = words[i].strip()
                if words[i] == "":
                    i += 1
                    continue
                length_in_pixels = (len(words[i]) + 1) * self.char_width
                remaining_pixels = width - length_in_pixels
                line = words[i]

                while remaining_pixels > 0:
                    i += 1
                    if i >= len(words):
                        break
                    length_in_pixels = (len(words[i]) + 1) * self.char_width
                    remaining_pixels -= length_in_pixels
                    if remaining_pixels < 0:
                        continue
                    else:
                        line += " " + words[i]

                line_array = [
                    line,
                    int(start) + 15,
                    int(len(line) / total_chars * total_frames) + int(start) + 15,
                ]
                start = int(len(line) / total_chars * total_frames) + int(start)
                lines.append(line_array)
                self.text_array.append(line_array)

        cap.release()
        print("Transcription complete")

    def extract_audio(self):
        print("Extracting audio")
        audio_path = f"./dataset/youtube/{self.laugh_slice.video_title}/temp_audio"
        os.makedirs(audio_path, exist_ok=True)
        audio_path = f"{audio_path}/audio.mp3"
        video = VideoFileClip(self.video_path)
        audio = video.audio
        audio.write_audiofile(audio_path)
        self.audio_path = audio_path
        print("Audio extracted")

    def extract_frames(self, output_folder):
        print("Extracting frames")
        cap = cv2.VideoCapture(self.video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        asp = width / height
        N_frames = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = frame[
                :,
                int(int(width - 1 / asp * height) / 2) : width
                - int((width - 1 / asp * height) / 2),
            ]

            for i in self.text_array:
                if N_frames >= i[1] and N_frames <= i[2]:
                    text = i[0]
                    text_size, _ = cv2.getTextSize(
                        text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
                    )
                    text_x = int((frame.shape[1] - text_size[0]) / 2)
                    text_y = int(height / 2)
                    cv2.putText(
                        frame,
                        text,
                        (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75,
                        (0, 0, 255),
                        2,
                    )
                    break

            cv2.imwrite(os.path.join(output_folder, str(N_frames) + ".jpg"), frame)
            N_frames += 1

        cap.release()
        print("Frames extracted")

    def create_video(self, output_video_path):
        print("Creating video")
        image_folder = f"./dataset/youtube/{self.laugh_slice.video_title}/frames"
        os.makedirs(image_folder, exist_ok=True)
        self.extract_frames(image_folder)

        images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
        images.sort(key=lambda x: int(x.split(".")[0]))

        frame = cv2.imread(os.path.join(image_folder, images[0]))
        height, width, layers = frame.shape

        clip = ImageSequenceClip(
            [os.path.join(image_folder, image) for image in images], fps=self.fps
        )
        audio = AudioFileClip(self.audio_path)
        clip = clip.set_audio(audio)
        clip.write_videofile(output_video_path)
        # shutil.rmtree(image_folder)
        # os.remove(os.path.join(os.path.dirname(self.video_path), "audio.mp3"))

    # ===============================
    # Utilities
    # ===============================

    def _time_to_seconds(self, t):
        h, m, s = map(float, t.split(":"))
        return h * 3600 + m * 60 + s

    def _handle_dirs(self):
        os.makedirs(f"{self.transcripts_path}/audio", exist_ok=True)
        os.makedirs(f"{self.transcripts_path}/frames", exist_ok=True)
        os.makedirs(f"{self.transcripts_path}/output", exist_ok=True)

    def run(self, output_path: str):
        self.extract_audio()
        self.transcribe_video()
        self.create_video(output_path)
