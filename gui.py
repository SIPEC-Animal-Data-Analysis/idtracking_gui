import numpy as np
import cv2
from time import sleep
from tqdm import tqdm
import matplotlib.pyplot as plt
import time

from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument('--filename',
			action='store',
			dest='filename',
			type=str,
			help='filename of the video to be processed (has to be a segmented one)')

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

args = parser.parse_args()


filename = args.filename
sink = './example_data/frames/' + filename + '_segmented/'
results_sink = args.results_sink
num_masks = args.num_masks

print('loading masks')
results = np.load('./example_data/masks/' + filename + '_SegResults.npy', allow_pickle=True)

# limit now to 1000 frames for demo purposes and limited masks
results = results[:500]
num_masks = 5

# load a couple of frames
fnames = []

print('loading frames')
for i in tqdm(range(0,len(results))):
    example_frame = np.load(sink + filename + '_frame_'+ str(i) +'.npy')
    fnames.append(example_frame)

# take random masks
idxs = np.random.randint(0,len(results),num_masks)

font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (512,512)
fontScale              = 500
fontColor              = (0,255,0)
lineType               = 20
cv2.namedWindow("output", cv2.WINDOW_NORMAL)        # Create window with freedom of dimensions
cv2.resizeWindow('output', 1024,1024)

res = []
stepsize=1
_idx = 0
idx = idxs[_idx]

#start timer for persistent saving
start_time = time.time()

print('starting GUI')
while(True):
    try:
        y1, x1, y2, x2 = results[idx]['rois'][0]
    except IndexError:
        idx=idx-1
 
    if(idx==idxs[_idx]):
        cv2.rectangle(fnames[idx],(x1,y1),(x2,y2),(0,255,0),3)
        cv2.putText(fnames[idx],'Mask: ' + str(_idx),(10,100), font, 4,(255,255,255),2,cv2.LINE_AA)

    cv2.imshow("output", fnames[idx])                            # Show image
    frameclick = cv2.waitKey(1) & 0xFF

    if(frameclick == ord('q')):
        break
    if(frameclick == ord('a')):
        while(True):
            if(cv2.waitKey(20) & 0xFF == ord('a')):
                break
            sleep(0.1)
    if(frameclick == ord('n')):
        idx+=stepsize
        cv2.putText(fnames[idx],'Mask: ' + str(_idx),(10,100), font, 4,(255,255,255),2,cv2.LINE_AA)
    if(frameclick == ord('p')):
        idx-=stepsize
        cv2.putText(fnames[idx],'Mask: ' + str(_idx),(10,100), font, 4,(255,255,255),2,cv2.LINE_AA)

    for j in range(1,5):
        if(frameclick == ord(str(j))):
            res.append([fnames[idx], results[idx], str(j)])
            _idx+=1
            try:
                idx = idxs[_idx]
            except IndexError:
                continue
    if(frameclick == ord('d')):
        res.append([fnames[idx], results[idx], 'd'])
        _idx+=1
        try:
            idx = idxs[_idx]
        except IndexError:
            continue


    if(_idx == len(idxs)):
        break
    if( (time.time() - start_time) % 300 < 0.055):
        print('writing data, do not interrupt!')
        np.save(results_sink + 'IDresults_' + filename + '.npy', res)
        print('done writing data')
    sleep(0.05)

cv2.destroyAllWindows()
print('writing data, do not interrupt!')
np.save(results_sink + 'IDresults_' + filename + '.npy', res)
print('done writing data')


