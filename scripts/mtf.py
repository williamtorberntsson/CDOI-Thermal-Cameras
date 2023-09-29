"""This file contains a script to calculate the MTF for a camera with a specific integration-time

The module contains the following functions:

- `CalculateNETD` - Compute the Step-response, LSF, MTF for a camera.
- `getCroppedImage` - Returns the cropped image for the MTF-calculations.
"""

import pathlib
import numpy as np
import glob
import cv2 as cv
import math
import matplotlib.pyplot as plt
from scipy.fft import fft
from scipy.optimize import curve_fit


def CalculateMTF(input_image_dir:str, output_dir:str, camera_name:str):
    """Compute the Step-response, LSF, MTF for a camera.

    Args:
        input_image_dir: A string-path to a folder which contains one .tif-file   
    """
    print("Running MTF calculations")
    
    mtf_path = str(pathlib.Path(input_image_dir) / 'MTF_*' / '*.tif')
    for mtf_image in glob.glob(mtf_path):
        MTF_image = mtf_image

    # Create point matrix get coordinates of mouse click on image
    img = cv.imread(str(MTF_image), cv.IMREAD_UNCHANGED)

    # Cropp image to wanted area
    cropped_image = getCroppedImage(img)

    # Set the cropped image to int32
    img_matrix = np.int32(cropped_image)
    print(f'dtype: {img_matrix.dtype}, shape: {img_matrix.shape}, min: {np.min(img_matrix)}, max: {np.max(img_matrix)}')


    # Plot att the step-responses for the image
    figure_name = 1
    for row in range(0,np.shape(img_matrix)[0]-1,40):

        step_response = img_matrix[row,:,0]/255
 
        # Calculate min threshold, i.e. when derivative is approximately zero.
        threashold_der = 0.01
        min_index = 0
        max_index = 0
        max_value = 0
        min_value = 0

        # Calculate the threasholds to when the step-response goes from low to high. 
        derivative_approx = math.inf
        for i in range(5,np.shape(step_response)[0]-5):
            derivative_approx = (step_response[i+5] - step_response[i-5])/((i+5)-(i-5))
            if derivative_approx > threashold_der:
                min_index = i
                min_value = step_response[i]
                break

        derivative_approx = math.inf
        for i in range(np.shape(step_response)[0]-6,5,-1):
            derivative_approx = (step_response[i+5] - step_response[i-5])/((i+5)-(i-5))
            if derivative_approx > threashold_der:
                max_index = i
                max_value = step_response[i]
                break
 
        # How many pixel it takes for the step-response to go from black to white
        pixels_to_change = max_index-min_index

        # Determine each pixel in the image
        dx = np.linspace(0,np.shape(step_response)[0],np.shape(step_response)[0])
        dx = np.transpose(dx)

        # Calculate line spread function (Derivative of step-responses)
        lsf = np.diff(step_response)/np.diff(dx)

        # Calculate MTF (Fouriertransform of the line spread function)
        mtf = fft(lsf)
        N = np.shape(mtf)[0] # Length of mtf
        f_n = np.linspace(0,1,round(N/2)) # Normalized spatial frequency
        mtf_n = mtf[0:round(N/2)]
        mtf_n = np.abs(mtf_n)
        mtf_n = [float(i)/max(np.abs(mtf_n)) for i in mtf_n] # Make sure we have real numbers

        # Approximate MTF curve to a 1 degree polynomial
        [a, b], _ = curve_fit(lambda x1,a,b: a*np.exp(b*x1),  f_n,  mtf_n)
        mtf_exponential_approximation = a * np.exp(b * f_n)
    
        # Plot the LSF and, MTF Step-response
        plt.figure(num=figure_name)
        figure_name += 1
        plt.plot(step_response)
        plt.plot(lsf)
        plt.plot(min_index,min_value,'oy')
        plt.plot(max_index,max_value,'oc')
        plt.fill_between([min_index, max_index],0,1, alpha=0.2)
        plt.title('Average Step-response and Line Spread Function for row ' +str(row) + ' for ' + str(camera_name))
        plt.legend(['Step-Response','Line Spread Function','Threashold min','Threashold max',str(str(pixels_to_change)+' Pixel to Change')])
        plt.xlabel('Pixel-number')
        plt.ylabel('Normalized pixel-value')

        plt.savefig(pathlib.Path(output_dir / f'mtf_lsf_step_row_{row}.png'))
        plt.figure(num=figure_name)
        figure_name += 1
        plt.plot(f_n, np.abs(mtf_n),'r')
        plt.plot(f_n, np.abs(mtf_exponential_approximation),'c')
        
        plt.title('Average MTF for row ' +str(row) + ' for ' + str(camera_name))
        plt.xlabel('Normalized Spatial Frequency')
        plt.legend(['MTF','Approximated exponential MTF \n mtf(f)= ' + str(np.round(a,1)) + '*exp(' + str(np.round(b,1)) + '*f_n)'])

        plt.savefig(pathlib.Path(output_dir / f'mtf_average_row_{row}.png'))
    plt.show()


def getCroppedImage(img:np.ndarray) -> np.ndarray:
    """Crop an image to be used for MTF-calculation

    Args:
        img: .tif-file to cropp to wanted size

    Returns:
        cropped_image: The cropped image which the user has chosen
    """

    print('Select an area in the image FROM black TO white!')
    # Select ROI
    selected_roi_image = cv.selectROI("select the area", img)
    
    # Crop image
    cropped_image = img[int(selected_roi_image[1]):int(selected_roi_image[1]+selected_roi_image[3]),
                        int(selected_roi_image[0]):int(selected_roi_image[0]+selected_roi_image[2])]
    
    # Display cropped image
    cv.waitKey(5)
    cv.destroyAllWindows()

    return cropped_image

