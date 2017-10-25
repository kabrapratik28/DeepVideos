import os
import random
from frame_extraction import frame_extractor
import cPickle

class datasets(object):
    def __init__(self, batch_size=64, val_split=0.005, test_split=0.005, heigth=64, width=64, DIR='../../data', output_filename='../../all_videos.txt', ):
        self.file_path = os.path.abspath(os.path.dirname(__file__))
        self.DIR = os.path.join(self.file_path,DIR)
        self.output_filename = os.path.join(self.file_path,output_filename)
        self.batch_size = batch_size
        self.flagged_activities = ['PlayingDaf', 'BodyWeightSquats', 'Nunchucks', 'ShavingBeard', 'SkyDiving']
        self.data = None
        self.frame_ext = frame_extractor(heigth=heigth,width=width)
        self.videos_to_text_file()
        self.load_problematic_videos()
        self.train_test_split(val_split,test_split)

    def load_problematic_videos(self):
        _frames_file = os.path.join(self.file_path, 'frames.pickle')
        _problem_videos_file = os.path.join(self.file_path, 'problematic_videos.pickle')
        with open(_frames_file, 'rb') as fp:
            short_frames = cPickle.load(fp)
        with open(_problem_videos_file, 'rb') as fp:
            problematic_videos = cPickle.load(fp)

        self.blacklist = set(short_frames + problematic_videos)

    def videos_to_text_file(self):
        with open(self.output_filename, "w") as a:
            for path, subdirs, files in os.walk(self.DIR):
               for filename in files:
                 f = os.path.join(path, filename)
                 a.write(str(f) + os.linesep)


    def train_test_split(self, split_test_data, split_validation_data):
        """
        split_test_data : '%' of test data to split between 0 to 1
        """
        data = {}
        unseen = []
        seen = []
        for line in open(self.output_filename):
            line = line.rstrip('\n')
            if line in self.blacklist:
                continue
            if any(substring in line for substring in self.flagged_activities):
                unseen.append(line)
            else:
                seen.append(line)

        datasize = len(seen)

        #Random Shuffle
        random.shuffle(seen)

        validation_index = int(datasize * split_validation_data)
        data['validation'] = seen[:validation_index]

        seen = seen[validation_index:]
        test_index = int(datasize * split_test_data)
        data['train'] = seen[test_index:]
        data['test'] = seen[:test_index]
        data['unseen'] = unseen

        self.data = data

    def train_next_batch(self,):
        """Returns lists of length batch_size.
        This is a generator function, and it returns lists of the
        entries from the supplied iterator.  Each list will have
        batch_size entries, although the final list may be shorter.
        """
        train_iter = iter(self.data['train'])
        while True:
            curr_batch = []
            while len(curr_batch) < self.batch_size:
                entry = None
                try:
                    entry = train_iter.next()
                except StopIteration:
                    # Shuffle data for next rollover ...
                    random.shuffle(self.data['train'])
                    train_iter = iter(self.data['train'])
                if entry != None:
                    curr_batch.append(entry)
            if curr_batch:
                yield self.frame_ext.get_frames(curr_batch)

    def fixed_next_batch(self,data_iter):
        is_done = False
        while True:
            curr_batch = []
            while len(curr_batch) < self.batch_size:
                entry = None
                try:
                    entry = data_iter.next()
                except StopIteration:
                    is_done = True
                    break
                if entry != None:
                    curr_batch.append(entry)
            if len(curr_batch)==self.batch_size:
                yield self.frame_ext.get_frames(curr_batch)
            if is_done:
                break

    def val_next_batch(self,):
        """
        Returns lists of length batch_size.
        This is a generator function, and it returns lists of the
        entries from the supplied iterator.  Each list will have
        batch_size entries, although the final list may be shorter.
        """
        val_iter = iter(self.data['validation'])
        return self.fixed_next_batch(val_iter)

    def test_next_batch(self,):
        """Returns lists of length batch_size.
        This is a generator function, and it returns lists of the
        entries from the supplied iterator.  Each list will have
        batch_size entries, although the final list may be shorter.
        """
        val_iter = iter(self.data['test'])
        return self.fixed_next_batch(val_iter)
