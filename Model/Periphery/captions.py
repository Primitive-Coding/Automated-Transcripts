import os
import shutil
import cv2


# Data
import pandas as pd
import numpy as np

import whisper


class Captions:
    def __init__(self, data: pd.DataFrame, paths: dict) -> None:
        self.data = data
        self.paths = paths

        self.text_array = []
        self.fps = 0
        self.char_width = 0

    def generate_transcripts(self):

        self.model = whisper.load_model()
