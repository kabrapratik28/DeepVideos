import skvideo.io
from os import sys, path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from datasets import batch_generator as bg

def test_aslan_data(aslan_log):
    d = bg.datasets()
    videos = [line.rstrip('\n') for line in open('../../all_videos.txt')]

    for video in videos:
        if 'aslan' in video.lower():
            try:
                videodata = skvideo.io.vread(video)
                n = videodata.shape[0]
                print("Checking video {} with frames {}".format(video, n))
                max_n = max(max_n, n)
                min_n = min(min_n, n)
                if n < 35:
                    aslan_log.write(video + '\n')

            except:
                aslan_log.write(video + '\n')

if __name__ == "__main__" and __package__ is None:
    aslan_log = open("aslan_log.txt", "a")
    test_aslan_data(aslan_log)
    aslan_log.close()
