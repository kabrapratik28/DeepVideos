import skvideo.io
import os
import sys
from batch_generator import datasets
import cPickle

d = datasets()
videos = [line.rstrip('\n') for line in open('../../all_videos.txt')]

max_n = 0
min_n = 99999999
frames = []
problematic_videos = []
i = 0
for video in videos:
    try:
        videodata = skvideo.io.vread(video)
        n = videodata.shape[0]
        print("Checking video {} with frames {}".format(i, n))
        max_n = max(max_n, n)
        min_n = min(min_n, n)
        if n < 35:
            frames.append(video)
        i += 1
    except:
        problematic_videos.append(video)



with open(r"frames.pickle", "wb") as output_file:
    cPickle.dump(frames, output_file)
with open(r"problematic_videos.pickle", "wb") as output_file:
    cPickle.dump(problematic_videos, output_file)
