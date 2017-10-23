import os

class DataSets(object):
    def __init__(self, DIR='../../data', output_filename='all_videos.txt', **kwargs):
        self.DIR = DIR
        self.output_filename = output_filename

    def videos_to_text_file(self):
        with open(self.output_filename, "w") as a:
            for path, subdirs, files in os.walk(self.DIR):
               for filename in files:
                 f = os.path.join(path, filename)
                 a.write(str(f) + os.linesep)


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

if __name__ == "__main__":
    a = DataSets()
    a.videos_to_text_file()
