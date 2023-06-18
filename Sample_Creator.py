from PIL import Image
import numpy as np
import sys
import cv2
import argparse
from pathlib import Path

def ai8x_normalize(img):
    """
    normalize the sample as it is done in the ai8x library
    """
    return img.sub(0.5).mul(256.).round().clamp(min=-128, max=127)

def convert(file, from_dir):
    """
    Python utility for converting an image to a 64x64 sampledata.h file
    compatible with the model's known answer test.
    """
    np.set_printoptions(threshold=sys.maxsize)
    
    if from_dir:
        src = np.load(file)
    else:
        src = file
    src = src.astype(np.uint8)
    
    #Get red channel from img
    red = src[0,:,:]
    red_f = red.ravel()

    #Get green channel from img
    green = src[1,:,:]
    green_f = green.ravel()

    #Get blue channel from img
    blue = src[2,:,:]
    blue_f = blue.ravel()

    arr_result = []

    # 0x00bbggrr
    for i in range(len(red_f)):
        result = red_f[i] | green_f[i]<<8 | blue_f[i]<<16
        arr_result.append((result))
        
    #convert list to numpy array
    out_arr_result = np.asarray(arr_result, dtype=np.uint32)

    #Write out data to the header file
    with open('sampledata_created.h', 'w') as outfile:
        outfile.write('#define SAMPLE_INPUT_0 { \\')
        outfile.write('\n')

        for i in range(len(out_arr_result)):
            if i==0:
                outfile.write('\t0x{0:08x},\t'.format((out_arr_result[i])))

            else :
                d = i%8
                if(d!=0):
                    outfile.write('0x{0:08x},\t'.format((out_arr_result[i])))
                else:
                    outfile.write('\\')
                    outfile.write('\n\t')
                    outfile.write('0x{0:08x},\t'.format((out_arr_result[i])))

        outfile.write('\\')
        outfile.write('\n')
        outfile.write('}')
        outfile.write('\n')

    print("FINISH")
    #sys.stdout.close()
    