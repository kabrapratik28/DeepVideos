import os
import random

class DataSets(object):
    def __init__(self, DIR='../../data', output_filename='all_videos.txt', **kwargs):
        self.DIR = DIR
        self.output_filename = output_filename
        self.flagged_activities = ['PlayingDaf', 'BodyWeightSquats', 'Nunchucks', 'ShavingBeard', 'SkyDiving']

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
        unseen = [line.rstrip('\n') for line in open(self.output_filename) if any(substring in line for substring in self.flagged_activities)]
        seen = [line.rstrip('\n') for line in open(self.output_filename) if any(substring in line for substring in self.flagged_activities)]
        validation_index = int(len(seen)* (split_test_data))
        seen = random.shuffle(seen)
        data['validation'] = seen[:validation_index]

        seen = seen[validation_index:]
        test_index = int(len(seen)* (split_test_data))
        data['train'] = seen[test_index:]
        data['test'] = seen[:test_index]
        data['unseen'] = unseen

        return data


    @staticmethod
    def batch_iterator(iterator, batch_size):
        """Returns lists of length batch_size.
        This is a generator function, and it returns lists of the
        entries from the supplied iterator.  Each list will have
        batch_size entries, although the final list may be shorter.
        """
        entry = True  # Make sure we loop once
        while entry:
            batch = []
            while len(batch) < batch_size:
                try:
                    entry = iterator.next()
                except StopIteration:
                    entry = None
                if entry is None:
                    # End of file
                    break
                batch.append(entry)
            if batch:
                yield batch


a = DataSets()
# a.videos_to_text_file()
videos = a.train_test_split(0.2, 0.1)
