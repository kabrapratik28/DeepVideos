import os
import random

class datasets(object):
    def __init__(self, DIR='../../data', output_filename='all_videos.txt', batch_size=64, **kwargs):
        self.DIR = DIR
        self.output_filename = output_filename
        self.batch_size = batch_size
        self.flagged_activities = ['PlayingDaf', 'BodyWeightSquats', 'Nunchucks', 'ShavingBeard', 'SkyDiving']
        self.data = None


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
        seen = [line.rstrip('\n') for line in open(self.output_filename) if not any(substring in line for substring in self.flagged_activities)]
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
                	random.shuffle(data['train'])
                    train_iter = iter(self.data['train'])
                if entry != None:
	                curr_batch.append(entry)
            if curr_batch:
                yield curr_batch

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
                yield curr_batch

	def val_next_batch(self,):
        """Returns lists of length batch_size.
        This is a generator function, and it returns lists of the
        entries from the supplied iterator.  Each list will have
        batch_size entries, although the final list may be shorter.
        """
        val_iter = iter(self.data['validation'])
        returns fixed_next_batch(val_iter)

    def test_next_batch(self,):
        """Returns lists of length batch_size.
        This is a generator function, and it returns lists of the
        entries from the supplied iterator.  Each list will have
        batch_size entries, although the final list may be shorter.
        """
        val_iter = iter(self.data['test'])
        returns fixed_next_batch(val_iter)

