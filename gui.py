import os
import time
from argparse import ArgumentParser
from time import sleep

import cv2
import numpy as np
from skimage.transform import rescale
from tqdm import tqdm
from glob import glob

parser = ArgumentParser()

# TODO: make whole thing OOP
# TODO: load previous annotation as default
# TODO: train masks with mattermask - if that, test the full pipleine with mattermask instead
# TODO: maybe also ask for some simple annotated behavior movies?
# TODO: indicate labelled animals
# TODO: prevent choosing frame too close to the interval ends
# -- meting with valerio - ask for pc access so i can setup stuff
''' other todos
lokaler slider anstatt arrows -> fuer videos (=/- 5 minuets )
flag - einfach / vs. hard

upload mode - > oder so

Masken und identification -> fps -> uebersichts data frame

menschenliches Gesicht ausschneiden???? —> extra interface point
-> starting point -> video mit menschlichen faces —> von sepp
'''

parser.add_argument('--filename',
					action='store',
					dest='filename',
					type=str,
					help='filename of the video to be processed (has to be a segmented one)')

parser.add_argument('--names',
					action='store',
					dest='names',
					type=str,
					help='Name of primates in order [1,2,3,4]')

parser.add_argument('--out_folder',
					action='store',
					dest='results_sink',
					type=str,
					default='./results/',
					help='folder where results should be saved')

parser.add_argument('--num_masks',
					action='store',
					dest='num_masks',
					type=int,
					default=40,
					help='number of masks to be labeled for this video')

parser.add_argument('--window_size',
					action='store',
					dest='window_size',
					type=int,
					default=1024,
					help='size of the GUI in pixels')

class WindowHandler:
	frames = None
	current_frame = None

	def __init__(self,
				 frames_path,
				 name_indicators,
				 filename,
				 results_sink,
				 masks,
				 num_masks,
				 stepsize,
				 window_size, ) -> None:

		super().__init__()
		self.masks = masks
		self.frames_path = frames_path
		self.name_indicators = name_indicators
		self.filename = filename
		self.results_sink = results_sink
		self.current_mask_focus = 0
		self.zoom = False
		self.stepsize = stepsize
		self.break_status = False
		self.num_masks = num_masks



		# start timer for persistent saving
		self.start_time = time.time()

		# opencv params

		self.window_name = "output"

		cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)  # Create window with freedom of dimensions
		cv2.resizeWindow(self.window_name, window_size, window_size)
		self.font = cv2.FONT_HERSHEY_SIMPLEX
		self.bottomLeftCornerOfText = (512, 512)
		self.fontScale = 500
		self.fontColor = (0, 255, 0)
		self.lineType = 20
		self.font_thickness = 1

		try:
			self.results = self.load_data()
			self.previous_frame_focus = 0
			print('Loading previous annotations')
		except FileNotFoundError:
			self.results = {}
			self.previous_frame_focus = None

		self.manual_mode = False
		self.current_mask = 0
		self.current_frame_focus = self.draw_random_frame()
		self.current_frame = self.current_frame_focus
		self.current_difficulty_flag = 'easy'
		self.mask_focus = 0


		self.mask_color_default_focus_frame = (125, 125, 125)
		self.mask_color_default = (75, 75, 75)
		self.mask_color_focus = (0, 255, 0)
		self.mask_color_labeled = (0, 0, 255)

		self.local_slider_window = 20
		self.local_slider_lower_window = self.current_frame - self.local_slider_window
		self.local_slider_higher_window = self.current_frame + self.local_slider_window

		# load frames

		self.frames_length = len(glob(frames_path + '*.npy'))
		self.frames = []
		self.frames_mult = 99999
		self.load_frames(0,
		                 # self.frames_length)
		                 500)
		# self.load_frames(max(self.local_slider_lower_window * self.frames_mult , 0),
		#                  min(self.local_slider_higher_window * self.frames_mult, self.frames_length - 1))

		# self.load_frames(max(self.current_frame - 1 , 0),
		#                  min(self.current_frame + 1, self.frames_length - 1))


		self.local_slider = cv2.createTrackbar("Local Slider", self.window_name,
											   self.local_slider_lower_window,
											   self.local_slider_higher_window, self.on_change_local)
		self.global_slider = cv2.createTrackbar("Global Slider", self.window_name,
												self.current_frame, self.frames_length - 1, self.on_change_global)

	def load_frames(self,
	                start,
	                end):
		for i in tqdm(range(start, end)):
			example_frame = np.load(self.frames_path + 'frame_' + str(i) + '.npy')
			# self.frames[i] = example_frame
			self.frames.append(example_frame)

	def on_change_local(self,
						int):
		self.current_frame = int
		cv2.setTrackbarPos("Local Slider", self.window_name, int)
		cv2.setTrackbarPos("Global Slider", self.window_name, int)

	def on_change_global(self,
						int):
		self.current_frame = int
		cv2.setTrackbarPos("Global Slider", self.window_name, int)
		if not self.local_slider_lower_window < int < self.local_slider_higher_window:
			self.local_slider_lower_window = int - self.local_slider_window
			self.local_slider_higher_window = int + self.local_slider_window
			# self.load_frames(max(self.current_frame - 1, 0),
			#                  min(self.current_frame + 1, self.frames_length - 1))
			cv2.setTrackbarMin("Local Slider", winname = self.window_name,
							minval = self.local_slider_lower_window)
			cv2.setTrackbarMax("Local Slider", winname = self.window_name,
							maxval = self.local_slider_higher_window)
		cv2.setTrackbarPos("Local Slider", self.window_name, int)


	def close(self):
		print('writing data, do not interrupt!')
		self.save_data()
		print('done writing data')
		cv2.destroyAllWindows()

	def save_data(self):
		np.save(self.results_sink + 'IDresults_' + self.filename + '.npy', self.results)

	def load_data(self):
		return np.load(self.results_sink + 'IDresults_' + self.filename + '.npy', allow_pickle=True).item()

	def clocked_save(self):
		# save data every minute
		if (time.time() - self.start_time) % 60 < 0.055:
			self.save_data()

	def mask_to_opencv(self,
					   frame,
					   mask,
					   color,
					   animal_id=None,
					   mask_id=None):
		cv2.rectangle(frame, (mask[1], mask[0]), (mask[3], mask[2]), color, 3)
		if animal_id:
			if mask_id==0:
				cv2.putText(frame, animal_id,
							(mask[1], mask[0]), self.font, 0.5, self.mask_color_labeled, self.font_thickness, cv2.LINE_AA)
			if mask_id==1:
				cv2.putText(frame, animal_id,
							(mask[3], mask[2]), self.font, 0.5, self.mask_color_labeled, self.font_thickness, cv2.LINE_AA)
			if mask_id==2:
				cv2.putText(frame, animal_id,
							(mask[1], mask[2]), self.font, 0.5, self.mask_color_labeled, self.font_thickness, cv2.LINE_AA)
			if mask_id==3:
				cv2.putText(frame, animal_id,
							(mask[3], mask[0]), self.font, 0.5, self.mask_color_labeled, self.font_thickness, cv2.LINE_AA)

	def display_mask(self,
					 mask_id,
					 current_mask
					 ):

		frame = self.frames[self.current_frame]
		is_labeled = None
		try:
			is_labeled = mask_id in self.results[self.current_frame]['results'].keys()
		except KeyError:
			pass
		# display focus mask in focus frame12
		if self.current_frame == self.current_frame_focus:
			if mask_id == self.current_mask_focus:
				self.mask_to_opencv(frame, current_mask, self.mask_color_focus)
			elif is_labeled:
				animal_id = self.results[self.current_frame]['results'][mask_id]
				self.mask_to_opencv(frame, current_mask, self.mask_color_labeled, animal_id=animal_id,
									mask_id=mask_id)
			else:
				self.mask_to_opencv(frame, current_mask, self.mask_color_default_focus_frame)
		else:
			if is_labeled:
				animal_id = self.results[self.current_frame]['results'][mask_id]
				self.mask_to_opencv(frame, current_mask, self.mask_color_labeled, animal_id=animal_id,
									mask_id=mask_id)
			else:
				self.mask_to_opencv(frame, current_mask, self.mask_color_default)

	# # indicate mask number
	# cv2.putText(self.frames[self.current_frame], 'Mask: ' + str(self.current_frame),
	#             (10, 100), self.font, 4, (255, 255, 255), 2, cv2.LINE_AA)

	def draw_random_frame(self,
						  window=50):
		draw = 0
		print(self.results.keys())
		while True:
			draw = np.random.randint(0, len(self.masks), 1)[0]
			print(draw)
			if window < draw < len(self.masks) - window and draw not in self.results.keys():
				print('True')
				break
			print('False')
		return draw

	def set_new_random_focus(self):
		self.current_mask += 1
		self.current_frame_focus = self.draw_random_frame()
		self.current_frame = self.current_frame_focus
		self.current_mask_focus = 0

		cv2.setTrackbarPos("Global Slider", self.window_name, self.current_frame)
		if not self.local_slider_lower_window < self.current_frame < self.local_slider_higher_window:
			self.local_slider_lower_window = self.current_frame - self.local_slider_window
			self.local_slider_higher_window = self.current_frame + self.local_slider_window
			# self.load_frames(max(self.current_frame - 1, 0),
			#                  min(self.current_frame + 1, self.frames_length - 1))
			cv2.setTrackbarMin("Local Slider", winname = self.window_name,
							minval = self.local_slider_lower_window)
			cv2.setTrackbarMax("Local Slider", winname = self.window_name,
							maxval = self.local_slider_higher_window)
		cv2.setTrackbarPos("Local Slider", self.window_name, self.current_frame)

	def set_focus(self,
	              focus_frame):
		# self.current_mask += 1
		print(str(focus_frame))
		self.current_frame = focus_frame
		# self.current_mask_focus = 0

		cv2.setTrackbarPos("Global Slider", self.window_name, self.current_frame)
		if not self.local_slider_lower_window < self.current_frame < self.local_slider_higher_window:
			self.local_slider_lower_window = self.current_frame - self.local_slider_window
			self.local_slider_higher_window = self.current_frame + self.local_slider_window
			# self.load_frames(max(self.current_frame - 1, 0),
			#                  min(self.current_frame + 1, self.frames_length - 1))
			cv2.setTrackbarMin("Local Slider", winname = self.window_name,
							minval = self.local_slider_lower_window)
			cv2.setTrackbarMax("Local Slider", winname = self.window_name,
							maxval = self.local_slider_higher_window)
		cv2.setTrackbarPos("Local Slider", self.window_name, self.current_frame)


	def display_frame(self):
		if self.zoom:
			img = self.frames[self.current_frame]
			y1, x1, y2, x2 = self.masks[self.current_frame]['rois'][self.current_mask_focus]
			center_x = float(x2 + x1) / 2.0
			center_y = float(y2 + y1) / 2.0

			# TODO: make relative
			masked_img = img[int(center_y - 200):int(center_y + 200),
						int(center_x - 200):int(center_x + 200)]

			# TODO: determine value or find best fixed
			rescaled_img = rescale(masked_img, 1.75, multichannel=True)
			cv2.imshow("output", cv2.cvtColor(rescaled_img.astype('float32'), cv2.COLOR_BGR2RGB))
		else:
			curr_img = self.frames[self.current_frame]
			cv2.putText(curr_img, 'Frame: ' + str(self.current_frame), (10, 200),
						self.font, 4, (255, 255, 255), self.font_thickness, cv2.LINE_AA)
			cv2.putText(curr_img, 'Mask: ' + str(len(self.results)+1), (10, 100),
						self.font, 4, (255, 255, 255), self.font_thickness, cv2.LINE_AA)
			cv2.imshow("output", cv2.cvtColor(curr_img, cv2.COLOR_BGR2RGB))
		return

	def display_all_indicators(self):
		for idx, indicator in enumerate(self.name_indicators.keys()):
			cv2.putText(self.frames[self.current_frame], self.name_indicators[indicator] + ' : ' + str(indicator+1),
						(10, 850 + 25*idx), self.font, 0.5, (255, 255, 255), self.font_thickness, cv2.LINE_AA)


	def display_all_keys(self):
		#TODO: explain 'b','w','t'
		##
		dist_1 = 200
		cv2.putText(self.frames[self.current_frame], 'p -- display previous mask',
					(dist_1, 850 + 0), self.font, 0.5, (255, 255, 255), self.font_thickness, cv2.LINE_AA)
		cv2.putText(self.frames[self.current_frame], 'b -- reset view to current mask',
					(dist_1, 850 + 25), self.font, 0.5, (255, 255, 255), self.font_thickness, cv2.LINE_AA)
		cv2.putText(self.frames[self.current_frame], 'm -- trigger manual mode on current frame',
					(dist_1, 850 + 50), self.font, 0.5, (255, 255, 255), self.font_thickness, cv2.LINE_AA)
		cv2.putText(self.frames[self.current_frame], 'w -- wrong mask (not a primate)',
					(dist_1, 850 + 75), self.font, 0.5, (255, 255, 255), self.font_thickness, cv2.LINE_AA)
		cv2.putText(self.frames[self.current_frame], 't -- too difficult to annotate',
					(dist_1, 850 + 100), self.font, 0.5, (255, 255, 255), self.font_thickness, cv2.LINE_AA)

		##
		dist_2 = 600
		cv2.putText(self.frames[self.current_frame], '. -- change mask to focus on forward',
					(dist_2, 850 + 0), self.font, 0.5, (255, 255, 255), self.font_thickness, cv2.LINE_AA)
		cv2.putText(self.frames[self.current_frame], ', -- change mask to focus on backward',
					(dist_2, 850 + 25), self.font, 0.5, (255, 255, 255), self.font_thickness, cv2.LINE_AA)
		cv2.putText(self.frames[self.current_frame], '= -- zoom in',
					(dist_2, 850 + 50), self.font, 0.5, (255, 255, 255), self.font_thickness, cv2.LINE_AA)
		cv2.putText(self.frames[self.current_frame], '- -- zoom out',
					(dist_2, 850 + 75), self.font, 0.5, (255, 255, 255), self.font_thickness, cv2.LINE_AA)

		pass

	def save_mask_result(self,
	                     result):
		if self.current_frame == self.current_frame_focus:
			try:
				self.results[self.current_frame]['frame']
			except KeyError:
				self.results[self.current_frame] = {
					'frame': self.frames[self.current_frame],
					'masks': self.masks[self.current_frame],
					}
			try:
				results = self.results[self.current_frame]['results']
				results[self.current_mask_focus] = result
			except KeyError:
				# first result indicates mask_id, second indicates animal id
				results = {self.current_mask_focus: result}
				self.results[self.current_frame]['results'] = results
			# change to next random frame if all masks labeled
			if self.current_mask_focus == len(self.masks[self.current_frame_focus]['rois']) - 1:
				print('setting new focus')
				self.set_new_random_focus()
				self.previous_frame_focus = 0
				self.current_difficulty_flag = 'easy'
			# otherwise next mask
			else:
				self.current_mask_focus += 1
				self.current_difficulty_flag = 'easy'


	def display_current_frame(self):
		# if masks available, display them
		try:
			current_masks = self.masks[self.current_frame]['rois']
			for mask_id, mask in enumerate(current_masks):
				self.display_mask(mask_id,
								  mask)
		except IndexError:
			pass
		self.display_frame()

		return

	def check_keys(self):
		frameclick = cv2.waitKey(1) & 0xFF
		# quit
		if frameclick == ord('q'):
			self.break_status = True
		if frameclick == ord('a'):
			while (True):
				if cv2.waitKey(20) & 0xFF == ord('a'):
					self.break_status = True
				sleep(0.1)
		# # next frame
		# if (frameclick == ord('n')):
		#     self.current_frame += self.stepsize
		#     cv2.putText(self.frames[self.current_frame], 'Mask: ' + str(self.current_mask), (10, 100),
		#                 self.font, 4, (255, 255, 255), 2, cv2.LINE_AA)
		# # previous frame
		# if (frameclick == ord('p')):
		#     self.current_frame -= self.stepsize
		#     cv2.putText(self.frames[self.current_frame], 'Mask: ' + str(self.current_mask), (10, 100),
		#                 self.font, 4, (255, 255, 255), 2, cv2.LINE_AA)

		# TODO: finish easy/hard flag
		if frameclick == ord('h'):
			self.current_difficulty_flag = 'hard'
		# back to previous focus
		if frameclick == ord('p') and self.previous_frame_focus is not None:
			if self.previous_frame_focus < len(list(self.results.keys())):
				self.current_frame = list(self.results.keys())[self.previous_frame_focus]
				self.previous_frame_focus += 1
			else:
				self.previous_frame_focus += 0
		# back to current focus
		if frameclick == ord('b'):
			self.set_focus(self.current_frame_focus)
		# zoom in
		if frameclick == ord('='):
			self.zoom = True
		# zoom out
		if frameclick == ord('-'):
			self.zoom = False
		# change mask focus increasing
		if frameclick == ord('.'):
			if self.current_mask_focus < len(self.masks[self.current_frame]['rois']) - 1:
				self.current_mask_focus += 1
		# change mask focus decreasing
		if frameclick == ord(','):
			if self.current_mask_focus > 0:
				self.current_mask_focus -= 1
		# TODO: implement
		# manual mode trigger
		if frameclick == ord('m'):
			self.manual_mode = not self.manual_mode
			if self.manual_mode:
				self.current_frame_focus = self.current_frame
			else:
				self.current_frame_focus = self.draw_random_frame()
				self.current_frame = self.current_frame_focus
		# skipping a mask
		# TODO: fix what is results
		if frameclick == ord('w'):
			self.save_mask_result('wrong_mask')
		if frameclick == ord('t'):
			self.save_mask_result('too_difficult')
		# labeling one of the primates
		for j in range(1, len(self.name_indicators)+1):
			if frameclick == ord(str(j)):
				# TODO: multiple results are in same FOV
				self.save_mask_result(self.name_indicators[j - 1])


	def check_num_results(self):
		if len(self.results) == self.num_masks:
			self.break_status = True
			print('labeled enough masks')
		return

	def update(self):
		self.display_current_frame()
		self.display_all_indicators()
		self.display_all_keys()
		self.check_keys()
		self.clocked_save()
		self.check_num_results()
		return self.break_status

Videos = {
	'video1' : '20180115T150502-20180115T150902_%T1',
	'video2' : '',
	'video3' : '',
	'video4' : '',
	}

def main():
	# parse arguments
	args = parser.parse_args()
	names = args.names
	names = names.split(',')
	name_indicators = {}
	for idx, el in enumerate(names):
		name_indicators[idx] = el

	base_path = '/Users/marksm/Desktop/cpFromBob/idtracking_gui/'
	filename = args.filename
	filename = Videos[filename]
	sink = base_path + filename + '/'
	results_sink = args.results_sink
	num_masks = args.num_masks
	window_size = args.window_size

	print('loading masks')
	masks = np.load(base_path + filename + '/SegResults.npy', allow_pickle=True)

	# create results folder if non-existing
	if not os.path.exists(results_sink):
		os.makedirs(results_sink)

	# limit now to 1000 frames for demo purposes and limited masks
	# end = len(masks)
	end = 500
	masks = masks[0:end]

	# load a couple of frames
	frames = []
	print('loading frames')
	frames_path = sink + 'frames/'

	stepsize = 1

	# init handler
	myhandler = WindowHandler(frames_path,
							  name_indicators,
							  filename,
							  results_sink,
							  masks,
							  num_masks,
							  stepsize,
							  window_size)

	while (True):
		break_status = myhandler.update()
		if break_status:
			myhandler.save_data()
			break
		sleep(0.05)

if __name__ == '__main__':
	main()
