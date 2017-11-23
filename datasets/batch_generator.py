import os
import random
from frame_extraction import frame_extractor
import cPickle

class datasets(object):
    def __init__(self, batch_size=32, val_split=0.05, test_split=0.05, height=64, width=64,
                 DIR='../../data', output_filename='../../all_videos.txt',
                 interval=1, custom_test_size=[160,210], dataset='UCF-101',time_frame=32):

        self.file_path = os.path.abspath(os.path.dirname(__file__))
        self.DIR = os.path.join(self.file_path,DIR,dataset)
        self.output_filename = os.path.join(self.file_path,output_filename)
        self.batch_size = batch_size
        self.data = None
        self.all_vids = []
        self.custom_test_size = custom_test_size
        self.interval = interval
        self.time_frame = time_frame
        self.frame_ext = frame_extractor(height=height,width=width,time_frame=time_frame)
        self.videos_to_text_file()
        self.load_problematic_videos()
        self.train_test_split(val_split,test_split)
        self.dataset = dataset


    def load_problematic_videos(self):
        _frames_file = os.path.join(self.file_path, 'frames.pickle')
        _problem_videos_file = os.path.join(self.file_path, 'problematic_videos.pickle')
        base_frame_names = []
        with open(_frames_file, 'rb') as fp:
            short_frames = cPickle.load(fp)
            for each_name in short_frames:
            	base_frame_names += [os.path.basename(each_name),]
        with open(_problem_videos_file, 'rb') as fp:
            problematic_videos = cPickle.load(fp)
            for each_name in short_frames:
            	base_frame_names += [os.path.basename(each_name),]

        self.blacklist = set(base_frame_names)

    def videos_to_text_file(self):
        with open(self.output_filename, "w") as a:
            for path, subdirs, files in os.walk(self.DIR):
               for filename in files:
                 f = os.path.join(path, filename)
                 a.write(str(f) + os.linesep)
        self.categories = set(os.listdir(self.DIR))
        print ("Categories took into consideration ... ",len(self.categories))

    def train_test_split(self, split_test_data, split_validation_data):
        """
        split_test_data : '%' of test data to split between 0 to 1
        """
        for line in open(self.output_filename):
            line = line.rstrip('\n')
            if os.path.basename(line) in self.blacklist:
                continue
            else:
                self.all_vids.append(line)

        train = []
        validation = []
        test = []

        for category in self.categories:
            all_data_in_category = [path for path in self.all_vids if category in path]
            # datasize = len(all_data_in_category)
            # validation_index = int(datasize * split_validation_data)
            # validation.extend(all_data_in_category[:validation_index])
            # all_data_in_category = all_data_in_category[validation_index:]
            # test_index = int(datasize * split_test_data)

            # last two video validation, prev two test and rest is training
            train.extend( all_data_in_category[:-4] )
            test.extend( all_data_in_category[-4:-2] )
            validation.extend( list( all_data_in_category[-2:]) )

        random.shuffle(train)
        # within that validation set take any 50 ! (if you wan fix 50 uncomment below line of seed set)
        # random.seed(28)
        # **************************************************************
        # DONOT REMOVE SHUFFLE ELSE IT GIVES DATA FROM ONE CATEGORY !!!
        # **************************************************************
        random.shuffle(validation)
        random.shuffle(test)
        data = {'train':train, 'validation':validation[:50], 'test':test[:50]}
        self.data = data

    def train_next_batch(self,):
        """
        Returns lists of length batch_size.
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
                except:
                    # Shuffle data for next rollover ...
                    random.shuffle(self.data['train'])
                    train_iter = iter(self.data['train'])
                if entry != None:
                    curr_batch.append(entry)
            if curr_batch:
                yield self.frame_ext.get_frames_with_interval_x(curr_batch, x= self.interval)

    def fixed_next_batch(self,data_iter):
        # Batch size less than threshold
        assert self.batch_size < 50

        is_done = False
        while True:
            curr_batch = []
            while len(curr_batch) < self.batch_size:
                entry = None
                try:
                    entry = data_iter.next()
                except:
                    is_done = True
                    break
                if entry != None:
                    curr_batch.append(entry)
            if len(curr_batch)==self.batch_size:
                yield self.frame_ext.get_frames_with_interval_x(curr_batch, x= self.interval, randomize=False)
            if is_done:
                break

    def val_next_batch(self,):
        val_iter = iter(self.data['validation'])
        return self.fixed_next_batch(val_iter)

    def test_next_batch(self,):
        val_iter = iter(self.data['test'])
        return self.fixed_next_batch(val_iter)

    def get_custom_test_data(self):
        new_frame_ext = frame_extractor(height=self.custom_test_size[0], width=self.custom_test_size[1],time_frame=self.time_frame)
        # 3 good videos
        # vids = ['v_BoxingSpeedBag_g18_c03','v_MilitaryParade_g15_c06','v_SalsaSpin_g21_c02']
        # tv = self.data['train'] + self.data['validation']
        new_test = []
        for category in self.categories:
            all_data_in_category = [path for path in self.all_vids if category in path]
            if len(all_data_in_category) > 5:
                rand_smpl = [all_data_in_category[i] for i in sorted(random.sample(xrange(len(all_data_in_category)), 5))]
                new_test.extend(rand_smpl)

        new_test = new_test [ : (len(new_test) / self.batch_size)*self.batch_size]
        while True:
            curr_batch = []
            train_iter = iter(new_test)
            while len(curr_batch) < self.batch_size:
                entry = None
                try:
                    entry = train_iter.next()
                except:
                    break
                if entry != None:
                    curr_batch.append(entry)
            if curr_batch:
                yield new_frame_ext.get_frames_with_interval_x(curr_batch, x= self.interval, randomize=False)
