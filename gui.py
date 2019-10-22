import os
import time
from argparse import ArgumentParser
from time import sleep
import cv2


import numpy as np
from skimage.transform import rescale
from tqdm import tqdm

parser = ArgumentParser()

# TODO: make whole thing OOP
# TODO: load previous annotation as default
# TODO: train masks with mattermask - if that, test the full pipleine with mattermask instead
# TODO: maybe also ask for some simple annotated behavior movies?
# TODO: indicate labelled animals
# TODO: prevent choosing frame too close to the interval ends
# -- meting with valerio - ask for pc access so i can setup stuff
''' other todos
masken die schon gelable sind -> kleiner text der indiziert welcher schon (andere färbe)
unten - > kleine indication
lokaler slider anstatt arrows -> fuer videos (=/- 5 minuets )
globaler slider
für jedes tier aehnlich viele Masken
flag - einfach / vs. hard

manual tag mode -> einschalten und Maske wechseln möglich -> grüne Masken durchwechseln

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

## look in results sink for already annotated data
# try:
#   annotations = (results_sink + 'IDresults_' + filename + '.npy')
#   print('Loading previous annotations')
# except FileNotFoundError:
#   continue

class WindowHandler:

    frames = None
    current_frame = None

    def __init__(self,
                 frames,
                 name_indicators,
                 filename,
                 results_sink,
                 masks,
                 num_masks,
                 stepsize,
                 window_size, ) -> None:

        super().__init__()
        self.masks = masks
        self.frames = frames
        self.name_indicators = name_indicators
        self.filename = filename
        self.results_sink = results_sink
        self.current_mask_focus = 0
        self.zoom = False
        self.stepsize = stepsize
        self.break_status = False
        self.num_masks = num_masks

        # take random masks
        self.random_mask_indexes = np.random.randint(0, len(self.masks), 1)

        self.current_mask = 0
        self.current_frame = self.random_mask_indexes[self.current_mask]

        self.mask_focus = 0

        # start timer for persistent saving
        self.start_time = time.time()

        # opencv params

        cv2.namedWindow("output", cv2.WINDOW_NORMAL)  # Create window with freedom of dimensions
        cv2.resizeWindow('output', window_size, window_size)
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.bottomLeftCornerOfText = (512, 512)
        self.fontScale = 500
        self.fontColor = (0, 255, 0)
        self.lineType = 20

        self.results = {}

    # TODO:awggw
    def load_frames(self):
        pass

    def close(self):
        print('writing data, do not interrupt!')
        self.save_data()
        print('done writing data')
        cv2.destroyAllWindows()

    def save_data(self):
        np.save(self.results_sink + 'IDresults_' + self.filename + '.npy', self.results)

    def clocked_save(self):
        if ((time.time() - self.start_time) % 300 < 0.055):
            self.save_data()

    def display_mask(self,
                     current_mask,
                     focus=False):
        if (self.current_frame == self.random_mask_indexes[self.current_mask]):
            # indicate mask
            cv2.rectangle(self.frames[self.current_frame],
                          (current_mask[1], current_mask[0]),
                          (current_mask[3], current_mask[2]),
                          (0, 255, 0), 3)
            # indicate mask number
            cv2.putText(self.frames[self.current_frame], 'Mask: ' + str(self.current_frame),
                        (10, 100), self.font, 4, (255, 255, 255), 2, cv2.LINE_AA)
        else:
            cv2.rectangle(self.frames[self.current_frame],
                          (current_mask[1], current_mask[0]),
                          (current_mask[3], current_mask[2]),
                          (125, 125, 125), 3)
        if(self.current_frame in self.results.keys()):
            pass


        # # TODO:
        # for res in own results:
        #     display masks

    def display_frame(self):
        if(self.zoom):
            img = self.frames[self.current_frame]
            y1, x1, y2, x2 = self.masks[self.current_frame]['rois'][self.current_mask_focus]
            center_x = float(x2 + x1) / 2.0
            center_y = float(y2 + y1) / 2.0

            # TODO: make relative
            masked_img = img[int(center_y - 200):int(center_y + 200),
                         int(center_x - 200):int(center_x + 200)]

            # TODO: determine value or find best fixed
            rescaled_img = rescale(masked_img, 2.5, multichannel=True)
            cv2.imshow("output", rescaled_img)
        else:
            curr_img = self.frames[self.current_frame]
            cv2.putText(curr_img, 'Frame: ' + str(self.current_frame), (10, 200),
                        self.font, 4, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow("output", curr_img)
        return

    def display_current(self):
        try:
            current_masks = self.masks[self.current_frame]['rois']
            for mask in current_masks:
                self.display_mask(mask)
            self.display_mask(self.masks[self.current_frame]['rois'][self.current_mask_focus],
                              focus=True)
        except KeyError:
            pass

        self.display_frame()


        # TODO: here in general: make a dict of usable keys and print them
        # cv2.putText(curr_img, name_indicators[0], (10, 900),
        #             self.font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        # indicate current frame

        return

    def check_keys(self):

        frameclick = cv2.waitKey(1) & 0xFF
        # quit
        if (frameclick == ord('q')):
            self.break_status = True
        if (frameclick == ord('a')):
            while (True):
                if (cv2.waitKey(20) & 0xFF == ord('a')):
                    self.break_status = True
                sleep(0.1)
        # next frame
        if (frameclick == ord('n')):
            self.current_frame += self.stepsize
            cv2.putText(self.frames[self.current_frame], 'Mask: ' + str(self.current_mask), (10, 100),
                        self.font, 4, (255, 255, 255), 2, cv2.LINE_AA)
        # previous frame
        if (frameclick == ord('p')):
            self.current_frame -= self.stepsize
            cv2.putText(self.frames[self.current_frame], 'Mask: ' + str(self.current_mask), (10, 100),
                        self.font, 4, (255, 255, 255), 2, cv2.LINE_AA)
        # zoom in
        if (frameclick == ord('=')):
            self.zoom = True

        # zoom out
        if (frameclick == ord('-')):
            self.zoom = False

        # change mask focus increasing
        if (frameclick == ord('.')):
            self.current_mask_focus += 1

        # change mask focus decreasing
        if (frameclick == ord(',')):
            self.current_mask_focus -= 1

        # TODO: implement
        # manual mode trigger
        # if (frameclick == ord('m')):
        #     self.

        for j in range(1, 5):
            if (frameclick == ord(str(j))):
                # TODO: multiple results are in same FOV
                self.results[self.current_frame] = {
                    'frame' : self.frames[self.current_frame],
                    'masks' : self.masks[self.current_frame],
                    'current_mask' : self.current_mask_focus,
                    'identity' : self.name_indicators[j-1],
                }
                self.current_mask += 1
                self.current_frame = self.random_mask_indexes[self.current_mask]
                # TODO: implement this, or rather sample new random spots instead of prefixed list
                # try:
                #     self.current_frame = idxs[_idx]
                # except IndexError:
                #     continue
        # skip a mask
        # # FIXME: index errors at end
        # if (frameclick == ord('d')):
        #     self.results.append([self.frames[idx], self.masks[idx], 'd'])
        #     _idx += 1
        #     try:
        #         idx = idxs[_idx]
        #     except IndexError:
        #         continue
        # pass

    def check_num_results(self):
        if(len(self.results) == self.num_masks):
            self.break_status = True
            print('labeled enough masks')
        return

    def update(self):
        self.display_current()
        self.check_keys()
        self.clocked_save()
        self.check_num_results()
        return self.break_status




# FIXME: end when all masks
    # # gone through all masks
    # if (_idx == len(idxs)):
    #     break



def main():
    # parse arguments
    args = parser.parse_args()
    names = args.names
    names = names.split(',')
    name_indicators = ['Key/Name:  ' + str(idx) + ' / ' + el for idx, el in enumerate(names)]

    filename = args.filename
    sink = './example_data/frames/' + filename + '_segmented/'
    results_sink = args.results_sink
    num_masks = args.num_masks
    window_size = args.window_size

    print('loading masks')
    masks = np.load('./example_data/masks/' + filename + '_SegResults.npy', allow_pickle=True)

    # create results folder if non-existing
    if not os.path.exists(results_sink):
        os.makedirs(results_sink)

    # limit now to 1000 frames for demo purposes and limited masks
    masks = masks[2000:2500]

    # load a couple of frames
    frames = []
    print('loading frames')
    for i in tqdm(range(2000,2500)):
        example_frame = np.load(sink + filename + '_frame_' + str(i) + '.npy')
        frames.append(example_frame)

    stepsize = 1

    # init handler
    myhandler = WindowHandler(frames,
                              name_indicators,
                              filename,
                              results_sink,
                              masks,
                              num_masks,
                              stepsize,
                              window_size)

    while (True):
        status = myhandler.update()
        if(status):
            myhandler.save_data()
            break
        sleep(0.05)


if __name__ == '__main__':
    main()
