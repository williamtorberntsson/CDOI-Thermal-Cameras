"""This file contains a script to calculate the thermal noise, i.e NEDT, for a camera.

The module contains the following functions:

- `CalculateNEDT(map_with_csv_files, camera_name, cropped_image_factor)` - Returns the NEDT-value for a camera.
- `Split_Integration_Time(map_with_csv_files)` - Paths to each NEDT folder for a camera and the different integrationtimes.
- `Get_cropped_csv_files(path_to_files, cropped_image_factor)` - Returns the 3-dim arrays with 20, respectively 25, and 30 degrees celcius.
- `crop_matrix_for_NEDT(matrix, cropped_image_factor)` - Returns a cropped matrix.
- `Crop_and_save_correct_matrix(csv_matrix_to_save_in, cropped_image_factor, path_to_the_csv_files)` - Compute the 3-dim matrix and crop it to wanted area.
- `Get_TemporalNoise_Response(matrix_with_cropped_csv_20, matrix_with_cropped_csv_25, matrix_with_cropped_csv_30)` - Returns one array for 20, respectively 25, and 30 degrees celcius for a specific area.
- `Get_NEDT_value(response_20, response_30, temporal_noise)` - Returns the NEDT-value for the camera from 3 arrays.



Examples:
    >>> from scripts import netd
    >>> netd.CalculateNEDT('.\Measurements\A70')
    'NEDT value for A70 with cropping factor 0.8 and integration time 3.75 Hz, 0.0 mK'
"""

import os
import glob
import pathlib
import numpy as np
import os
import math


def CalculateNEDT(map_with_csv_files:str, camera_name:str, cropped_image_factor:float) -> tuple[np.ndarray, np.ndarray]:
    """Compute and return the NEDT-value for a camera.

    Args:
        map_with_csv_files:  A folder with a name on this form - NameOfTheCamera_xx_Hz. Where cameraname could be A70 and xx is the integration time.
        camera_name: Name of the camera to perform evaluations on
        cropped_image_factor: Centered percentage of image to crop and perform calculations on

    Returns:
        header: The text containing which camera it is, and which integration time the NEDT-value has been calculated on
        values: A number representing the NEDT-value
    """
    print("Running NEDT calculations")

    header = []
    values = []
    paths, int_time = Split_Integration_Time(map_with_csv_files)

    for i in range(0,len(int_time)):

        path_to_files = paths[i]

        try:
            header = np.append(header, f"NEDT value for {camera_name} with cropping factor {cropped_image_factor} and integration time {int_time[i]} Hz")
        except: 
            print("Folder does not exist") 
            
        try:
            matrix_with_cropped_csv_20, matrix_with_cropped_csv_25, matrix_with_cropped_csv_30 = Get_cropped_csv_files(path_to_files, cropped_image_factor)
        except: 
            print("Files could not be averaged or cropped")

        try:    
            response_20, response_30, temporal_noise = Get_TemporalNoise_Response(matrix_with_cropped_csv_20, matrix_with_cropped_csv_25, matrix_with_cropped_csv_30)
        except:
            print("Could not calculate Temporal noise and response of numpy arrays")  

        try:    
            value = Get_NEDT_value(response_20, response_30, temporal_noise)
            print(f"NEDT calculations done for camera {camera_name} with cropping factor {cropped_image_factor} and integration time {int_time[i]} Hz")
            values = np.append(values, f" {value} mK")
        except:
            print(f"Could not calculate NEDT value for camera {camera_name} with integration time {int_time[i]} Hz")        
            values = np.append(values," nan mK")

    return header, values


def Split_Integration_Time(map_with_csv_files):
    """
    Args:
        map_with_csv_files (str):  Original file which the user sends in into the function with all the NEDT folders

    Returns:
       all_path (array): Individual paths to a specific integration time
       int_time (array): A list containing the different integration times for the camera
    """

    nedt_all_integration_times = str(pathlib.Path(map_with_csv_files) / 'NEDT_')

    int_time_list = []
    int_time = []
    all_paths = []

    for folder in glob.glob(nedt_all_integration_times + '*'):
        split_name_folder = folder.split("_")

        if split_name_folder[1].replace(",", ".") not in int_time:
            int_time = np.append(int_time, split_name_folder[1].replace(",", "."))

            for int_time_folder in glob.glob(nedt_all_integration_times + split_name_folder[1] + '*'):
                int_time_list = np.append(int_time_list, int_time_folder)
            all_paths.append(int_time_list)
            int_time_list = []
    
    return all_paths, int_time


def Get_cropped_csv_files(path_to_files, cropped_image_factor):
    """Compute and return the 3 cropped matrices for 20, 25, and 30 degrees celsius.

    Args:
        path_to_files (array): Path to the folder with NEDT-files 
        cropped_image_factor (float): Centered percentage of image to crop and perform calculations on

    Returns:
        matrix_with_cropped_csv_30 (array): Cropped matrix with the values from the black-body measurement with 30 deg
        matrix_with_cropped_csv_25 (array): Cropped matrix with the values from the black-body measurement with 25 deg
        matrix_with_cropped_csv_20 (array): Cropped matrix with the values from the black-body measurement with 20 deg
    """

    paths_to_folders_with_20deg = []
    paths_to_folders_with_25deg = []
    paths_to_folders_with_30deg = []

    for temperature_folders in path_to_files:
        name_folder = temperature_folders.split("_")
        # See if the folder contains the files with 20, 25, or 30 degrees
        if name_folder[-3] == '20':
                paths_to_folders_with_20deg = [temperature_folders]
        if name_folder[-3] == '25':
                paths_to_folders_with_25deg = [temperature_folders]
        if name_folder[-3] == '30':
                paths_to_folders_with_30deg = [temperature_folders]  

    csv_matrix_20_deg = []
    csv_matrix_25_deg = []
    csv_matrix_30_deg = []

    # Read the csv-files to a matrix and crop it to wanted area.
    matrix_with_cropped_csv_20 = Crop_and_save_correct_matrix(csv_matrix_20_deg, cropped_image_factor, paths_to_folders_with_20deg)
    matrix_with_cropped_csv_25 = Crop_and_save_correct_matrix(csv_matrix_25_deg, cropped_image_factor, paths_to_folders_with_25deg)
    matrix_with_cropped_csv_30 = Crop_and_save_correct_matrix(csv_matrix_30_deg, cropped_image_factor, paths_to_folders_with_30deg)
    
    return matrix_with_cropped_csv_20, matrix_with_cropped_csv_25, matrix_with_cropped_csv_30


def Crop_and_save_correct_matrix(csv_matrix_to_save_in, cropped_image_factor, path_to_the_csv_files):
    """Compute the 3-dim matrix and crop it to wanted area.

    Args:
        csv_matrix_to_save_in (array): A matrix to save the matrix in.
        cropped_image_factor (float): Centered percentage of image to crop and perform calculations on
        path_to_the_csv_files (array): A path to where arrays are saved.

    Returns:
        csv_matrix_to_save_in (array): Matrix which the values in each csv-file has been appended to.
    """
    count = len(os.listdir(path_to_the_csv_files[0]))

    for column in range(0, count):
        try:
            # Read file for Macbooks
            path_csv_1 = str(path_to_the_csv_files[0]) + '/' + str(os.listdir(path_to_the_csv_files[0])[column].decode())
            csv_file = np.loadtxt(open(path_csv_1, "rb"), delimiter = ",")

        except:
            # Read files for windows and linux
            path_csv_1 = str(path_to_the_csv_files[0]) + '/' + str(os.listdir(path_to_the_csv_files[0])[column])
            csv_file = np.loadtxt(open(path_csv_1, "rb"), delimiter = ",")

        # Crop the image to obtain wanted area in matrix.
        matrix_with_cropped_csv = crop_matrix_for_NEDT(csv_file, cropped_image_factor)
        matrix_size = np.shape(matrix_with_cropped_csv) # Save the current matrix-size

        # Append the new calculated matrix to the csv_matrix
        csv_matrix_to_save_in = np.append(csv_matrix_to_save_in, matrix_with_cropped_csv)
        
    # Reshape the matrix to correct dimensions    
    csv_matrix_to_save_in = np.reshape(csv_matrix_to_save_in,(matrix_size[0],matrix_size[1], count))

    return csv_matrix_to_save_in      
   

def crop_matrix_for_NEDT(matrix, cropped_image_factor):
    """Crop an image to be used for NEDT

    Args:
        matrix (array): a matrix that will be cropped
        cropped_image_factor (float): Centered percentage of image to crop and perform calculations on

    Returns:
        cropped_matrix (array): Returns one matrix that is cropped to the specification of NEDT.
    """

    matrix_width = np.shape(matrix)[0]
    matrix_height = np.shape(matrix)[1]

    height = math.floor(cropped_image_factor * matrix_height)    # set cropped image height
    width = math.floor(cropped_image_factor * matrix_width)       # set cropped image width

    # makes a centered crop of the image
    cropped_matrix = matrix[math.floor(matrix_width/2) - math.floor(width/2) : math.floor(matrix_width/2) + math.floor(width/2), math.floor(matrix_height/2) - math.floor(height/2) : math.floor(matrix_height/2) + math.floor(height/2)]
    
    return cropped_matrix


def Get_TemporalNoise_Response(matrix_with_cropped_csv_20, matrix_with_cropped_csv_25, matrix_with_cropped_csv_30):
    """Compute and return a desired area which the NEDT-value will be calculated on.

    Args:
        matrix_with_cropped_csv_20 (array) : Matrix with values for 20 degrees celsius
        matrix_with_cropped_csv_25 (array) : Matrix with values for 25 degrees celsius
        matrix_with_cropped_csv_30 (array) : Matrix with values for 30 degrees celsius

    Returns:
        response_20 (array): Response at 20 degrees celsius
        response_30 (array): Response at 30 degrees celsius
        temporal_noise (array): Temporal noise at 25 degrees celsius
    """
    response_20 = np.zeros((np.shape(matrix_with_cropped_csv_20)[0],np.shape(matrix_with_cropped_csv_20)[1]))
    response_30 = np.zeros((np.shape(matrix_with_cropped_csv_30)[0],np.shape(matrix_with_cropped_csv_30)[1]))
    temporal_noise = np.zeros((np.shape(matrix_with_cropped_csv_25)[0],np.shape(matrix_with_cropped_csv_25)[1]))

    # Calculate the average response of the csv-files with 20 degrees celsius
    for matrix in range(0,np.shape(matrix_with_cropped_csv_20)[2]):
        response_20 += matrix_with_cropped_csv_20[:,:,matrix]
    response_20 /= np.shape(matrix_with_cropped_csv_20)[2]

    # Standard deviation with formula
    temporal_noise_average = np.mean(matrix_with_cropped_csv_25, axis=2)
    for matrix in range(0,np.shape(matrix_with_cropped_csv_25)[2]):
        temporal_noise += (np.abs(matrix_with_cropped_csv_25[:,:,matrix]-temporal_noise_average))**2
    temporal_noise /= np.shape(matrix_with_cropped_csv_25)[2]
    temporal_noise = np.sqrt(temporal_noise)
   
    # Calculate the average response of the csv-files with 30 degrees celsius
    for matrix in range(0,np.shape(matrix_with_cropped_csv_30)[2]):
        response_30 += matrix_with_cropped_csv_30[:,:,matrix]
    response_30 /= np.shape(matrix_with_cropped_csv_30)[2]

    return response_20, response_30, temporal_noise


def Get_NEDT_value(response_20, response_30, temporal_noise):
    """Calculate the NEDT-value for one camera.

    Args:
        response_20 (array) : response at 20 degrees
        response_30 (array) : response at 30 degrees
        temporal_noise (array) : standard deviation of all pixels at 128 frames.

    Returns:
        nedt_value (float): Returns the calculated NEDT-value for the camera
    """
    responsivity_matrix = (response_30 - response_20)/10 # unit counts/degree

    m_NEDT = temporal_noise/responsivity_matrix

    # averaging over all pixels not np.inf
    NEDT_value_kelvin = m_NEDT[m_NEDT != np.inf].mean()

    nedt_value = np.round(NEDT_value_kelvin*1000)
    return nedt_value 