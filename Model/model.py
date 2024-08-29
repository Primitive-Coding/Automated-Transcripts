# Periphery
# from Model.Periphery

# Sources
from Model.Sources.youtube import YouTubeVideo

from Model.Functions.get_transcriptions import (
    use_amazon_transcriptions,
    use_azure_transcriptions,
    use_google_transcriptions,
    use_ibm_transcriptions,
    use_whisper_transcriptions,
)


from Model.Networks.llm import LocalLLM

import librosa


class Model:
    def __init__(self, url: str) -> None:

        self.url = url
        self.video = YouTubeVideo(url, log=True)
        self.video.download_youtube_video_info()
        # self.llm = LocalLLM(model_name="gpt-neo")

        ls = self.video.get_laughter_slices()

        self.video.export_demo_clip(ls[4], "./demo-clips/temp.mp4")
        # print(ls[4].data)
        # self.video.export_all_clips()

        # print(loudest_slices[0].data)
        # self.video.plot_all_laughter_slices(laughter_slices, highlighted_slice=1)
        # self.llm.display_q_and_a("What is the capital of France?")

    def test(self):
        training_data = self.video.get_training_data()

        test_text = training_data["text"].iloc[0]

        t_encode = self.video.encode_transcript(test_text)
        t_decode = self.video.encode_transcript(t_encode, decode=True)

    def get_training_data(self):
        audio_features = self.merge_audio_features_with_text(False)
        features = [
            {key: d[key] for key in ["mfcc", "encoded"] if key in d}
            for d in audio_features
        ]
        return features

    def get_input_shape(self):
        audio_features = self.get_training_data()

        shape = audio_features[0]["mfcc"].shape
        return shape

    def merge_audio_features_with_text(self, return_dict: bool = True):
        """
        return_dict: bool
            If True, returns a dictionary where the values are lists. Ex: -> {"A": [], "B": []}
            If False, returns a list of dictionaries. Ex: -> [{"A": "", "B": ""}]

        """
        training_data = self.video.get_training_data()
        audio_features = []

        if return_dict:
            audio_features = {
                "file": [],
                "mfcc": [],
                "text": [],
                "encoded": [],
            }

            for i, row in training_data.iterrows():

                mfcc = self.video.get_chunk_audio_features(row["file"])
                text = row["text"]
                e_text = self.video.encode_transcript(text)

                audio_features["file"].append(row["file"])
                audio_features["mfcc"].append(mfcc)
                audio_features["text"].append(text)
                audio_features["encoded"].append(e_text)

        else:
            audio_features = []
            for i, row in training_data.iterrows():
                mfcc = self.video.get_chunk_audio_features(row["file"])
                text = row["text"]
                e_text = self.video.encode_transcript(text)
                data = {
                    "file": row["file"],
                    "mfcc": mfcc,
                    "text": text,
                    "encoded": e_text,
                }
                audio_features.append(data)

        return audio_features
