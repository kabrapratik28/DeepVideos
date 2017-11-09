import pytube
from pytube import YouTube
import os

class YoutubeFetcher(object):

    def __init__(self, num_videos, by_class = False, class_ids = None):
        self.num_videos = num_videos
        if by_class == True:
            if class_ids is None:
                raise ValueError("When by class is true you must pass a list of class ids")
        self.by_class = by_class
        self.class_ids = class_ids
        self.load_files()

    def load_files(self,):

        video_count = 0
        video_files = []
        with open('original/train_partition.txt') as fp:
            for line in fp:
                link, class_ = line.split()
                if self.by_class == False:
                    video_files.append(link)
                    video_count += 1
                else:
                    if any(int(cl) in self.class_ids for cl in class_.split(',')):
                        video_files.append(link)
                        video_count += 1
                if video_count == self.num_videos:
                    break
        self.video_files = video_files

    def download(self, ):

        count = 1
        for link in self.video_files:
            print("Downloading file {} of {}".format(count, len(self.video_files)))
            try:
		output_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "../../data/sports-1m")
                YouTube(link).streams.filter(subtype='mp4', only_video=True).last().download(output_path=output_path)
            except:
                print("Could'nt download {}".format(link))

            count += 1
def main():
    class_ids=[412,122,60,165,383]
    ytf = YoutubeFetcher(20, by_class=False, class_ids= None )
    ytf.download()

if __name__ == "__main__":
    main()
