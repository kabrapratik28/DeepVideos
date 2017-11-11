import random
import skvideo.io
import cv2
import numpy as np
import os
from moviepy.editor import *

class frame_extractor():
	def __init__(self,height=64, width=64, time_frame=32, dir_to_save=os.path.join(os.path.abspath(os.path.dirname(__file__)), "../../output/")):
		self.height = height
		self.width = width
		self.time_frame = time_frame
		#self.count = 0
		self.dir_to_save = dir_to_save

	def image_processing(self,X):
		X = (X - 127.5) / 127.5
		return X

	def image_postprocessing(self,X):
		X = (X * 127.5) + 127.5
		return X

	def get_frames_with_interval_x(self, list_video_filenames, x, randomize=True):
		train_X = []
		train_y = []
		for each_filename in list_video_filenames:
			try:
				video_data = skvideo.io.vread(each_filename)
				N, H, W, C = video_data.shape
				max_frame_number = N - ((self.time_frame + 1) * x)
				if max_frame_number<=0:   # very short video ! 
					continue
				frame_index = 0
				if max_frame_number>=1 and randomize == True:
					frame_index = random.randint(0,max_frame_number)
				data = video_data[frame_index : frame_index+(self.time_frame + 1) * x : x]
				frames = []
				for each_frame in data:
					resized_image = cv2.resize(each_frame, (self.width,self.height))
					frames.append(resized_image)
				frames = np.array(frames)
				X = frames[ 0 : self.time_frame]
				y = frames[ 1 : self.time_frame+1]
				train_X.append(X)
				train_y.append(y)
			except RuntimeError:
				print("Error in batch iterator, skipping video")
		train_X = self.image_processing(np.array(train_X))
		train_y = self.image_processing(np.array(train_y))
		return train_X, train_y, list_video_filenames

	def generate_output_video(self, frames, filenames):

		frames = self.image_postprocessing(frames)
		no_videos = frames.shape[0]
		no_frames = frames.shape[1]
		for i in range(no_videos):
			cur_video = np.array([frames[i][j] for j in range(no_frames)])
			filename = os.path.splitext(os.path.basename(filenames[i]))[0]
			skvideo.io.vwrite(os.path.join(self.dir_to_save, filename + '.mp4'), cur_video)
			#self.count += 1

	def generate_gif_videos(self, input_file_path, output_file_path, speed):
		video_freeze = (VideoFileClip(input_file_path).speedx(speed))
		video_freeze.write_gif(output_file_path)
