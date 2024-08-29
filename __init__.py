# from Model.Periphery.video import Video
from Model.model import Model

from Model.train_model import train

from Model.Sources.youtube import YouTubeVideo
from Model.Periphery.captions import Captions

from VideoTranscriptions.video_transcriber import VideoTranscriber


def test():
    url = "https://www.youtube.com/watch?v=zKUpf1Vx0vs"
    y = YouTubeVideo(url)
    y.convert_to_short()
    # v = VideoTranscriber(l_slices[3], paths)
    # v.run()
    # c = Captions(l_slices[0].data, paths)


if __name__ == "__main__":

    test()
    # url = "https://www.youtube.com/watch?v=zKUpf1Vx0vs"

    # m = Model(url)

    # train(m)
