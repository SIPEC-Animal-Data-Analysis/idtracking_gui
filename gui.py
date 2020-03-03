import sys
from joblib import Parallel, delayed

sys.path.extend(['./venv/lib/python3.7/site-packages'])
import os
import time
from argparse import ArgumentParser
from time import sleep

import cv2
import numpy as np
from skimage.transform import rescale
import skvideo.io
from tqdm import tqdm
from glob import glob

import skimage.color
import skimage.io
import skimage.transform

from distutils.version import LooseVersion

import pickle

crop = True
# highres is 500

crop = 500

parser = ArgumentParser()

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

parser.add_argument('--species',
                    action='store',
                    dest='species',
                    type=str,
                    help='define the species to annotate primate/mouse')



def resize(image, output_shape, order=1, mode='constant', cval=0, clip=True,
           preserve_range=False, anti_aliasing=False, anti_aliasing_sigma=None):
    """A wrapper for Scikit-Image resize().

    Scikit-Image generates warnings on every call to resize() if it doesn't
    receive the right parameters. The right parameters depend on the version
    of skimage. This solves the problem by using different parameters per
    version. And it provides a central place to control resizing defaults.
    """
    if LooseVersion(skimage.__version__) >= LooseVersion("0.14"):
        # New in 0.14: anti_aliasing. Default it to False for backward
        # compatibility with skimage 0.13.
        return skimage.transform.resize(
            image, output_shape,
            order=order, mode=mode, cval=cval, clip=clip,
            preserve_range=preserve_range, anti_aliasing=anti_aliasing,
            anti_aliasing_sigma=anti_aliasing_sigma)
    else:
        return skimage.transform.resize(
            image, output_shape,
            order=order, mode=mode, cval=cval, clip=clip,
            preserve_range=preserve_range)


def resize_image(image, min_dim=None, max_dim=None, min_scale=None, mode="square"):
    # Keep track of image dtype and return results in the same dtype
    image_dtype = image.dtype
    # Default window (y1, x1, y2, x2) and default scale == 1.
    h, w = image.shape[:2]
    window = (0, 0, h, w)
    scale = 1
    padding = [(0, 0), (0, 0), (0, 0)]
    crop = None

    if mode == "none":
        return image, window, scale, padding, crop

    # Scale?
    if min_dim:
        # Scale up but not down
        scale = max(1, min_dim / min(h, w))
    if min_scale and scale < min_scale:
        scale = min_scale

    # Does it exceed max dim?
    if max_dim and mode == "square":
        image_max = max(h, w)
        if round(image_max * scale) > max_dim:
            scale = max_dim / image_max

    # Resize image using bilinear interpolation
    if scale != 1:
        image = resize(image, (round(h * scale), round(w * scale)),
                       preserve_range=True)

    # Need padding or cropping?
    if mode == "square":
        # Get new height and width
        h, w = image.shape[:2]
        top_pad = (max_dim - h) // 2
        bottom_pad = max_dim - h - top_pad
        left_pad = (max_dim - w) // 2
        right_pad = max_dim - w - left_pad
        padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
        image = np.pad(image, padding, mode='constant', constant_values=0)
        window = (top_pad, left_pad, h + top_pad, w + left_pad)
    elif mode == "pad64":
        h, w = image.shape[:2]
        # Both sides must be divisible by 64
        assert min_dim % 64 == 0, "Minimum dimension must be a multiple of 64"
        # Height
        if h % 64 > 0:
            max_h = h - (h % 64) + 64
            top_pad = (max_h - h) // 2
            bottom_pad = max_h - h - top_pad
        else:
            top_pad = bottom_pad = 0
        # Width
        if w % 64 > 0:
            max_w = w - (w % 64) + 64
            left_pad = (max_w - w) // 2
            right_pad = max_w - w - left_pad
        else:
            left_pad = right_pad = 0
        padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
        image = np.pad(image, padding, mode='constant', constant_values=0)
        window = (top_pad, left_pad, h + top_pad, w + left_pad)
    elif mode == "crop":
        # Pick a random crop
        h, w = image.shape[:2]
        y = random.randint(0, (h - min_dim))
        x = random.randint(0, (w - min_dim))
        crop = (y, x, min_dim, min_dim)
        image = image[y:y + min_dim, x:x + min_dim]
        window = (0, 0, min_dim, min_dim)
    else:
        raise Exception("Mode {} not supported".format(mode))
    return image.astype(image_dtype), window, scale, padding, crop


def mold_image(img):
    image, window, scale, padding, crop = resize_image(
        img[:, :, :],
        min_dim=2048,
        min_scale=2048,
        max_dim=2048,
        mode="square")
    return image


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
                 window_size, ):

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
        self.current_frame_focus = 1
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

        # load frames --- old frames

        #self.overall_frames = len(glob(frames_path + '*.npy'))
        #self.frame_buffer = min(10000,self.overall_frames)
        #self.frame_batches = int(float(self.overall_frames)/float(self.frame_buffer))
        #self.frame_current_batch = 0
        
        ## int((self.frame_current_batch - 1) * self.frame_buffer)
        #self.frames_start = 0
        #self.frames_length = int( (self.frame_current_batch+1) * self.frame_buffer)
        #self.frames = []
        #self.frames_mult = 99999
        #print('loading frames')
        #self.load_frames(0, self.frames_length)
        
        # --- old frames end
        self.load_frames(0, 0)
        self.overall_frames = len(self.frames)

        self.local_slider = cv2.createTrackbar("Local Slider", self.window_name,
                                               self.local_slider_lower_window,
                                               self.local_slider_higher_window, self.on_change_local)
        self.global_slider = cv2.createTrackbar("Global Slider", self.window_name,
                                                self.current_frame, self.overall_frames - 1, self.on_change_global)

    def load_frames(self,
                    start,
                    end):
        print('loading frames')        
        # frames as npy version
#        for i in tqdm(range(start, end)):
#            example_frame = np.load(self.frames_path + 'frame_' + str(i) + '.npy')
#            # self.frames[i] = example_frame
#            self.frames.append(example_frame)


        #TODO: remove hardcoding
        segment = int(self.filename.split('_')[-1][-1])
        vidname = self.filename.split('_1')[0]
        basepath = '/media/nexus/storage1/swissknife_data/primate/raw_videos/2018_2/'
        vid = basepath + vidname + '.mp4'
        # frames from mp4
        videodata = skvideo.io.vread(vid, as_grey=False)
        videodata = videodata[(segment-1) * 10000:segment * 10000]
        results = Parallel(n_jobs=40,
            max_nbytes=None,
            backend='multiprocessing',
            verbose=40)(delayed(mold_image)(image) for image in videodata)
        self.frames = videodata


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
            cv2.setTrackbarMin("Local Slider", winname=self.window_name,
                               minval=self.local_slider_lower_window)
            cv2.setTrackbarMax("Local Slider", winname=self.window_name,
                               maxval=self.local_slider_higher_window)
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
            if mask_id == 0:
                cv2.putText(frame, animal_id,
                            (mask[1], mask[0]), self.font, 0.5, self.mask_color_labeled, self.font_thickness,
                            cv2.LINE_AA)
            if mask_id == 1:
                cv2.putText(frame, animal_id,
                            (mask[3], mask[2]), self.font, 0.5, self.mask_color_labeled, self.font_thickness,
                            cv2.LINE_AA)
            if mask_id == 2:
                cv2.putText(frame, animal_id,
                            (mask[1], mask[2]), self.font, 0.5, self.mask_color_labeled, self.font_thickness,
                            cv2.LINE_AA)
            if mask_id == 3:
                cv2.putText(frame, animal_id,
                            (mask[3], mask[0]), self.font, 0.5, self.mask_color_labeled, self.font_thickness,
                            cv2.LINE_AA)

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

    def adjust_trackbar(self):

        # check whether current frame outside focus
        if not self.frame_current_batch == self.check_batchnum(self.current_frame):
            print('reloading frames')
            self.frame_current_batch = self.check_batchnum(self.current_frame)
            self.load_frames(self.frame_buffer * self.frame_current_batch, self.frame_buffer *
                             (self.frame_current_batch + 1))

        self.current_mask_focus = 0

        cv2.setTrackbarPos("Global Slider", self.window_name, self.current_frame)
        if not self.local_slider_lower_window < self.current_frame < self.local_slider_higher_window:
            self.local_slider_lower_window = self.current_frame - self.local_slider_window
            self.local_slider_higher_window = self.current_frame + self.local_slider_window
            # self.load_frames(max(self.current_frame - 1, 0),
            #                  min(self.current_frame + 1, self.frames_length - 1))
            cv2.setTrackbarMin("Local Slider", winname=self.window_name,
                               minval=self.local_slider_lower_window)
            cv2.setTrackbarMax("Local Slider", winname=self.window_name,
                               maxval=self.local_slider_higher_window)
        cv2.setTrackbarPos("Local Slider", self.window_name, self.current_frame)

    def check_batchnum(self,
                       frame):

        for i in range(0,self.frame_batches):
            if int(self.frame_buffer * i) < frame < int(self.frame_buffer * (i+1)):
                return i
        return -1

    def set_new_regular_focus(self,
                              interval=500):
        self.current_mask += 1
        self.current_frame_focus = self.current_frame_focus + 200
        self.current_frame = self.current_frame_focus
        
        #TODO: nicer
        breakval = True
        while(breakval):
            try:
                self.masks[self.current_frame]['rois'][0]
                breakval = False
            except IndexError:
                self.current_frame_focus = self.current_frame_focus + 200
                self.current_frame = self.current_frame_focus

        if self.current_frame > 9500:
            self.break_state = True
        
        self.adjust_trackbar()

    def set_new_random_focus(self):
        self.current_mask += 1
        self.current_frame_focus = self.draw_random_frame()
        self.current_frame = self.current_frame_focus
        self.current_mask_focus = 0

        self.adjust_trackbar()

    def set_focus(self,
                  focus_frame):
        # self.current_mask += 1
        print(str(focus_frame))
        self.current_frame = focus_frame
        # self.current_mask_focus = 0

        self.adjust_trackbar()

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
            rescaled_img = rescale(masked_img, 1.75, multichannel=True)[crop:-crop, :]
            cv2.imshow("output", cv2.cvtColor(rescaled_img.astype('float32'), cv2.COLOR_BGR2RGB))
        else:
            curr_img = self.frames[self.current_frame][crop:-crop, :]
            cv2.putText(curr_img, 'Frame: ' + str(self.current_frame), (1000, 800),
                        self.font, 4, (255, 255, 255), self.font_thickness, cv2.LINE_AA)
            cv2.putText(curr_img, 'Mask: ' + str(len(self.results) + 1), (1000, 900),
                        self.font, 4, (255, 255, 255), self.font_thickness, cv2.LINE_AA)
            cv2.imshow("output", cv2.cvtColor(curr_img, cv2.COLOR_BGR2RGB))
        return

    def display_all_indicators(self):
        for idx, indicator in enumerate(self.name_indicators.keys()):
            cv2.putText(self.frames[self.current_frame], self.name_indicators[indicator] + ' : ' + str(indicator + 1),
                        (10, 850 + 25 * idx), self.font, 0.5, (255, 255, 255), self.font_thickness, cv2.LINE_AA)

    def display_all_keys(self):
        # TODO: explain 'b','w','t'
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
        cv2.putText(self.frames[self.current_frame], 'h -- difficult to annotate (hard to see from current frame)',
                    (dist_1, 850 + 100), self.font, 0.5, (255, 255, 255), self.font_thickness, cv2.LINE_AA)
        cv2.putText(self.frames[self.current_frame], 't -- too difficult to annotate',
                    (dist_1, 850 + 125), self.font, 0.5, (255, 255, 255), self.font_thickness, cv2.LINE_AA)

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
                results[self.current_mask_focus] = result + '_' + self.current_difficulty_flag
            except KeyError:
                # first result indicates mask_id, second indicates animal id
                results = {self.current_mask_focus: result + '_' + self.current_difficulty_flag}
                self.results[self.current_frame]['results'] = results
            # change to next random frame if all masks labeled
            if self.current_mask_focus == len(self.masks[self.current_frame_focus]['rois']) - 1:
                print('setting new focus')
                self.set_new_regular_focus()
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
        try:
            self.display_frame()
        except IndexError:
            self.break_status = True

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
        # next frame
        if frameclick == ord('k'):
            self.current_frame += self.stepsize
            cv2.putText(self.frames[self.current_frame], 'Mask: ' + str(self.current_mask), (10, 100),
                        self.font, 4, (255, 255, 255), 2, cv2.LINE_AA)
            self.adjust_trackbar()

        # previous frame
        if frameclick == ord('j'):
            self.current_frame -= self.stepsize
            cv2.putText(self.frames[self.current_frame], 'Mask: ' + str(self.current_mask), (10, 100),
                        self.font, 4, (255, 255, 255), 2, cv2.LINE_AA)
            self.adjust_trackbar()


        # next frame
        if frameclick == ord('i'):
            self.current_frame += int(self.stepsize*5)
            cv2.putText(self.frames[self.current_frame], 'Mask: ' + str(self.current_mask), (10, 100),
                        self.font, 4, (255, 255, 255), 2, cv2.LINE_AA)
            self.adjust_trackbar()

        # previous frame
        if frameclick == ord('u'):
            self.current_frame -= int(self.stepsize*5)
            cv2.putText(self.frames[self.current_frame], 'Mask: ' + str(self.current_mask), (10, 100),
                        self.font, 4, (255, 255, 255), 2, cv2.LINE_AA)
            self.adjust_trackbar()


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
            self.current_frame_focus = self.current_frame
            if self.current_frame_focus in self.results.keys():
                del self.results[self.current_frame_focus]

        # skipping a mask
        # TODO: fix what is results
        if frameclick == ord('w'):
            self.save_mask_result('wrong_mask')
        if frameclick == ord('t'):
            self.save_mask_result('too_difficult')
        # labeling one of the primates
        for j in range(1, len(self.name_indicators) + 1):
            if frameclick == ord(str(j)):
                # TODO: multiple results are in same FOV
                self.save_mask_result(self.name_indicators[j - 1])

    def check_num_results(self):
        if len(self.results) == self.num_masks:
            self.break_status = True
            print('labeled enough masks')
        return

    def update(self):
        try:
            self.display_current_frame()
            # self.display_all_indicators()
            # self.display_all_keys()
            self.check_keys()
            self.clocked_save()
            self.check_num_results()
        except (FileNotFoundError, IndexError):
            self.break_status = True
        return self.break_status


# videos_primate = {
#     'video1': '20180115T150502-20180115T150902_%T1',
#     'video2': '20180124T115800-20180124T122800b_%T1',
#     'video3': '20180131T135402-20180131T142501_%T1',
#     'video4': '20180202T140159-20180202T143159_%T1',
#     }

videos_primate = [
    #'20180131T135402-20180131T142501_%T1_1',
    #'20180131T135402-20180131T142501_%T1_2',
    #'20180124T115800-20180124T122800b_%T1_3',
    #'20180124T115800-20180124T122800b_%T1_4',
    #'20180202T140159-20180202T143159_%T1_1',
    #'20180202T140159-20180202T143159_%T1_2',
    #'20180131T135402-20180131T142501_%T1_3',
    #'20180131T135402-20180131T142501_%T1_4',
    #'20180124T115800-20180124T122800b_%T1_1',
    #'20180124T115800-20180124T122800b_%T1_2',
    #'20180202T140159-20180202T143159_%T1_3',
    #'20180202T140159-20180202T143159_%T1_4',
    '20180115T150502-20180115T150902_%T1_1',
    # '20180115T150502-20180115T150902_%T1_2',
    # '20180115T150502-20180115T150902_%T1_3',
    # '20180115T150502-20180115T150902_%T1_4',
]

videos_primate = [
    '20180126T145419-20180126T145619_%T1_1',
]

videos_mice = {
    'video1': 'Animal1234 Day1',
    'video2': 'Animal5678 Day1',
    'video3': 'Animal1234 Day2',
    'video4': 'Animal5678 Day2',
    'video5': 'Animal1234 Day3postswim',
    'video6': 'Animal1234 Day1',
    }

import gc
import joblib
import time
from multiprocessing import Process
import concurrent.futures

def load_mask(video_path):
    gc.disable()
    with open(video_path + 'SegResults.pkl', 'rb') as handle:
        masks = pickle.load(handle)
        # masks = joblib.load(handle, mmap_mode="r")
    gc.enable()
    return masks

def load_mask_parallel(video_path):
    gc.disable()
    with open(video_path + 'SegResults.pkl', 'rb') as handle:
        masks = pickle.load(handle)
        # masks = joblib.load(handle, mmap_mode="r")
    gc.enable()
    return masks

def main():
# parse arguments
    args = parser.parse_args()
    species = args.species
    names = args.names
    results_sink = args.results_sink
    num_masks = 100
    window_size = args.window_size
    names = names.split(',')
    name_indicators = {}
    for idx, el in enumerate(names):
        name_indicators[idx] = el

    base_path = '/media/nexus/storage1/swissknife_data/primate/inference/segmentation_highres_multi/'
    base_path = '/media/nexus/storage1/swissknife_data/primate/inference/segmentation_new/2018_2/'
    if not os.path.exists(results_sink):
        os.makedirs(results_sink)

    future = None
    executor = None
    myhandler = None

    for video_id, filename in enumerate(videos_primate):

        if myhandler:
            del myhandler
        # check num masks
        start = time.time()
        video_path = base_path + filename + '/'

        if executor:
            preload_masks = future.result()
            masks = preload_masks
            executor.shutdown(wait=False)
        else:
            masks = load_mask(video_path)
        print('loading mask took', time.time() - start)
        # if video complete go to next video
        # res = np.load(results_sink + video + '.npy')
        # if not len(res) < num_masks:
        #     continue

        frames_path = video_path + 'frames/'
        stepsize = 100

        print('initiating handler')
        # init handler
        myhandler = WindowHandler(frames_path,
                                  name_indicators,
                                  filename,
                                  results_sink,
                                  masks,
                                  num_masks,
                                  stepsize,
                                  window_size)



        # p1 = Process(target=load_mask, args=('hello',))
        # p1.start()
        # try:
        #     print('preloading masks')
        #     #ThreadPoolExecutor
        #     with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
        #         future = executor.submit(load_mask, base_path + videos_primate[idx+1] + '/')
        # except IndexError:
        #     pass

        # try:
        #     executor = concurrent.futures.ThreadPoolExecutor(max_workers=5)
        #     future = executor.submit(load_mask, base_path + videos_primate[video_id + 1] + '/')
        # except IndexError:
        #     pass

        breaking = True
        while breaking:
            break_status = myhandler.update()
            if break_status:
                print('saving data, dont interrupt')
                myhandler.save_data()
                print('data saved, good to interrupt')
                cv2.destroyAllWindows()
                breaking=False
                break
            sleep(0.05)

        return

if __name__ == '__main__':
    main()
