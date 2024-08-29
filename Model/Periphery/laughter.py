import pandas as pd


class Timestamp:
    def __init__(self, ts: str) -> None:
        self.timestamp = ts.split(".")[0]
        self.ms = self._time_to_milliseconds(ts)

    # ===============================
    # Utilities
    # ===============================
    # Function to convert HH:MM:SS.MMM to milliseconds
    def _time_to_milliseconds(self, t) -> int:
        h, m, s = t.split(":")
        s, ms = s.split(".")
        return (int(h) * 3600 + int(m) * 60 + int(s)) * 1000 + int(ms)


class Laughter:
    def __init__(self, df: pd.DataFrame, path, video_title: str) -> None:
        self.data = df
        self.start = Timestamp(self.get_start())
        self.end = Timestamp(self.get_end())
        self.path = path
        _split = path.split("/")
        self.file_name = _split[-1]
        self.file_index = self.file_name.split(".")[0][-1]
        self.video_title = video_title

    def __repr__(self) -> pd.DataFrame:
        return repr(self.data)

    def get_slice_duration(self):

        start = self.data["start"].iloc[0]
        end = self.data["end"].iloc[-1]
        start_ms = self._time_to_milliseconds(start)
        end_ms = self._time_to_milliseconds(end)
        duration = end_ms - start_ms
        return duration

    # ===============================
    # Get Attributes
    # ===============================
    def get_start(self):
        return self.data["start"].iloc[0]

    def get_end(self):
        return self.data["end"].iloc[-1]

    def get_average_volume(self):
        return self.data["volume"].mean()

    # ===============================
    # Get Attributes
    # ===============================

    # ===============================
    # Utilities
    # ===============================
    # Function to convert HH:MM:SS.MMM to milliseconds
    def _time_to_milliseconds(self, t) -> int:
        h, m, s = t.split(":")
        s, ms = s.split(".")
        return (int(h) * 3600 + int(m) * 60 + int(s)) * 1000 + int(ms)
