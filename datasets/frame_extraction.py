import random
import skvideo.io
import cv2
import numpy as np

class frame_extractor():
	def __init__(self,heigth=64, width=64, time_frame=32):
		self.heigth = heigth
		self.width = width
		self.time_frame = time_frame

	def get_frames(self, list_video_filenames):
		train_X = []
		train_y = []
		for each_filename in list_video_filenames:
			video_data = skvideo.io.vread(each_filename)
			N, H, W, C = video_data.shape
			max_frame_number = N - (self.time_frame+1)
			frame_index = 0 
			if max_frame_number>=1:
				frame_index = random.randint(0,max_frame_number)
			data = video_data[frame_index : frame_index+self.time_frame+1]
			frames = []
			for each_frame in data:
				resized_image = cv2.resize(each_frame, (self.heigth,self.width))
				frames.append(resized_image)
			frames = np.array(frames)
			X = frames[ 0 : self.time_frame]
			y = frames[ 1 : self.time_frame+1]
			train_X.append(X)
			train_y.append(y)
		train_X = np.array(train_X)
		train_y = np.array(train_y)
		return train_X, train_y
    
	def get_frames_with_interval_x(self, list_video_filenames, x=2):
		train_X = []
		train_y = []
		for each_filename in list_video_filenames:
			video_data = skvideo.io.vread(each_filename)
			N, H, W, C = video_data.shape
			max_frame_number = N - ((self.time_frame + 1) * x)
			frame_index = 0 
			if max_frame_number>=1:
				frame_index = random.randint(0,max_frame_number)
			data = video_data[frame_index : frame_index+(self.time_frame + 1) * x : x]
			frames = []
			for each_frame in data:
				resized_image = cv2.resize(each_frame, (self.heigth,self.width))
				frames.append(resized_image)
			frames = np.array(frames)
			X = frames[ 0 : self.time_frame]
			y = frames[ 1 : self.time_frame+(1*x)]
			train_X.append(X)
			train_y.append(y)
		train_X = np.array(train_X)
		train_y = np.array(train_y)
		return train_X, train_y 