#!/usr/bin/python

import os
import argparse
import csv
import shutil
import glob
from pathlib import Path
from nedt import CalculateNEDT
from mtf import CalculateMTF
from temperature import CalculateStdTemperature
from distortion import CalculateDistortion

NEWLINE = '#' * 120

def main():
    inputs = parse_args()

    input_path = inputs.input_image_dir
    camera_name, output_path = extract_camera_name_output_path(input_path, inputs.output_dir, inputs.dist_new_dist_coeff)
    std_temp, nedt, mtf, dist = check_valid_directory(Path(input_path), inputs.input_temp_dir, output_path, inputs.remove_output_dir, camera_name)

    if inputs.calc_std_temp:
        calculate_and_save_std_dev_temperature(inputs.input_temp_dir, output_path, camera_name, std_temp)

    if inputs.calc_nedt:
        calculate_and_save_nedt(input_path, output_path, camera_name, nedt, inputs.cropped_image_factor)

    if inputs.calc_mtf:
        calculate_and_save_mtf(input_path, output_path, camera_name, mtf)

    if inputs.calc_dist:
        calculate_and_save_dist(input_path, output_path, camera_name, dist, inputs.dist_save_img, inputs.dist_save_data, 
                                inputs.dist_chessboard_columns, inputs.dist_chessboard_rows, inputs.dist_chessboard_tile_size,
                                inputs.dist_new_dist_coeff)

    if inputs.all_metrics:
        calculate_and_save_std_dev_temperature(inputs.input_temp_dir, output_path, camera_name, std_temp)
        calculate_and_save_nedt(input_path, output_path, camera_name, nedt, inputs.cropped_image_factor)
        calculate_and_save_mtf(input_path, output_path, camera_name, mtf)
        calculate_and_save_dist(input_path, output_path, camera_name, dist, inputs.dist_save_img, inputs.dist_save_data, 
                                inputs.dist_chessboard_columns, inputs.dist_chessboard_rows, inputs.dist_chessboard_tile_size,
                                inputs.dist_new_dist_coeff)
    
    write_to_csv_file(NEWLINE, [], output_path)


def calculate_and_save_std_dev_temperature(input_temp_dir:str, output_path:str, camera_name:str, std_temp:bool):
    '''Performs calculations of standard deviation for camera temperature 
    measurements in selected input file and appends results to output .csv file, 
    if a vaild path to data is provided.

    Args:
        input_temp_dir: Directory to file where temperature measurements 
        are stored
        output_path: Selected directory for storage of output file
        camera_name: Name of the camera to perform evaluations on
        std_temp: Bool variable previously set to True if a vaild path for 
        measured temperature data is provided
    '''    

    while True:
        if not std_temp:
            print("No vaild path for measured temperature data. Continuing with calcluations.")
            break
        extracted_headers_std_temp, extracted_values_std_temp = CalculateStdTemperature(input_temp_dir, camera_name)
        write_to_csv_file(extracted_headers_std_temp, extracted_values_std_temp, output_path)
        break 


def calculate_and_save_nedt(input_path:str, output_path:str, camera_name:str, nedt:bool, cropped_image_factor:float):
    '''Performs NEDT calculations for images in selected input directory and appends results to output .csv file, 
    if a vaild path to data is provided.

    Args:
        input_path: Selected directory to input files needed to perform calculations
        output_path: Selected directory for storage of output file
        camera_name: Name of the camera to perform evaluations on
        nedt: Bool variable Previously set to True if a vaild path for NEDT data is provided
        cropped_image_factor: Centered and cropped percentage of each image used for NEDT calculations
    '''    
    while True:
        if not nedt:
            print("No vaild path for NEDT data. Continuing with calcluations.")
            break
        extracted_headers_nedt, extracted_values_nedt = CalculateNEDT(input_path, camera_name, cropped_image_factor)
        write_to_csv_file(extracted_headers_nedt, extracted_values_nedt, output_path)
        break 


def calculate_and_save_mtf(input_path:str, output_path:str, camera_name:str, mtf:bool):
    '''Performs MTF calculations for images in selected input directory and stores results in output directory, 
    if a vaild path to data is provided.

    Args:
        input_path: Selected directory to input files needed to perform calculations
        output_path: Selected directory for storage of output file
        camera_name: Name of the camera to perform evaluations on
        mtf: Bool variable previously set to True if a vaild path for MTF data is provided
    '''  
    while True:
        if not mtf:
            print("No vaild path for MTF data. Continuing with calcluations.")
            break 
        CalculateMTF(input_path, output_path, camera_name)
        break


def calculate_and_save_dist(input_path:str, output_path:str, camera_name:str, dist:bool, save_images:bool, save_data:bool, columns:int, rows:int, tile_size:float, d_mod:list):
    '''Performs distortion calculations for images in selected input directory and stores results in output directory, 
    if a vaild path to data is provided.

    Args:
        input_path: Selected directory to input files needed to perform calculations
        output_path: Selected directory for storage of output file
        camera_name: Name of the camera to perform evaluations on
        dist: Bool variable previously set to True if a vaild path for distortion data is provided
    '''  
    while True:
        if not dist and not d_mod:
            print("No vaild path for distortion data. Continuing with calcluations.")
            break
        CalculateDistortion(input_path, output_path, camera_name, save_images, save_data, columns, rows, tile_size, d_mod)
        break

def extract_camera_name_output_path(input_path:str, output_dir:str, d_mod:list):
    '''Extracts camera name from input directory and creates output directory.

    Args:
        input_path: Selected parsed arg for directory to input files needed to perform calculations
        output_dir: Selected parsed arg for output directory for csv-file with derived parameters
        d_mod: Distortion parameter modificataion arg, indicating camera name needs to be extracted specially  
    Returns:
        output_path: Created directory for storage of output file with camera name as folder name
        camera_name: Name of the camera to perform evaluations on
    '''
    if d_mod:
        camera_name = Path(Path(input_path).parent).name
    else:
        camera_name = Path(input_path).name
    output_path = Path(output_dir) / camera_name
    return camera_name, output_path

def check_valid_directory(input_path:str, input_temp_path:str, output_path:str, clear_output:bool, camera_name:str):
    '''Checks if input directory is vaild and contains correct paths to data for calculating metrics.

    Args:
        input_path: Selected directory to input files needed to perform calculations
        input_temp_path: Directory to file where temperature measurements are stored
        output_path: Selected directory for storage of output file
        camera_name: Name of the camera to perform evaluations on

    Returns:
        std_temp: Bool set to True if input folder for camera contains properly stored data 
        for calculations of temperature standard deviations
        nedt: Bool set to True if input folder for camera contains properly stored data for calculations of NEDT
        mtf: Bool set to True if input folder for camera contains properly stored data for calculations of MTF 
        dist: Bool set to True if input folder for camera contains properly stored data for calculations of distortion
        
    '''
    if not Path(input_path).exists():
        print(f"{input_path} is not a valid input directory. \nPlease enter an exsisting input path.")
        exit()

    if not output_path.exists():
        os.makedirs(output_path)
        write_to_csv_file(NEWLINE, [], output_path)
    else:
        if clear_output:
            print(f"Removing already stored data for {camera_name} at {output_path} and starting new calculations")
            shutil.rmtree(output_path)
            os.makedirs(output_path)
            write_to_csv_file(NEWLINE, [], output_path)

    std_temp = False
    nedt = False
    mtf = False
    dist = False
    if glob.glob(str(input_temp_path)):
        std_temp = True
    if glob.glob(str(input_path / 'NEDT_*')):
        nedt = True
    if glob.glob(str(input_path / 'MTF_*')):
        mtf = True
    if glob.glob(str(input_path / 'Distortion_*')):
        dist = True
    
    return std_temp, nedt, mtf, dist


def write_to_csv_file(header, value, output_path:str):
    '''Appends computed values to output .csv file.

    Args:
        header: Description of which metric has been calculated for which camera
        value: Value of the metric beeing calculated
        output_path: Selected directory for storage of output file
    '''
    if any(header):
        with open(output_path / 'output.csv', 'a', encoding='UTF8') as f:
            writer = csv.writer(f, quoting=csv.QUOTE_NONE, escapechar='\\')
            if not any(value):
                writer.writerow([header])
            else:
                for i in range(len(header)):
                    writer.writerow([header[i], value[i]])
            writer.writerow('')


def parse_args():
    parser = argparse.ArgumentParser(description="Computes metrics for input sequence of thermal images",
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    required = parser.add_argument_group('required')

    required.add_argument('-i', '--input_image_dir', type = str, required=True,
    help = "Input directory to folder file containing .tif-files")
    
    parser.add_argument('-o', '--output_dir', type=str, default='output',
    help = "Output directory for csv-file with derived parameters")

    parser.add_argument('-rmo', '--remove_output_dir', action='store_true',
    help = "Clear output directory for csv-file with derived parameters before creating new")
    
    parser.add_argument('-it', '--input_temp_dir', type=str,
    help = "Input directory to .txt file containing temperature measurements")

    parser.add_argument('-std', '--calc_std_temp', action='store_true', 
    help = "Calulate standard deviation for temperature measurements in .txt file")

    parser.add_argument('-nedt', '--calc_nedt', action='store_true',
    help = "Calculate NEDT-value for a camera with a specific integration time in a folder")

    parser.add_argument('-crop', '--cropped_image_factor', type=float, default=0.5,
    help = "Desired factor to crop images for calculating NEDT")
    
    parser.add_argument('-mtf', '--calc_mtf', action='store_true',
    help = "Calculate the MTF for a camera")

    parser.add_argument('-a', '--all_metrics', action='store_true',
    help = "Calculate all metrics for a camera")

    # Distortion parse arguments
    parser.add_argument('-dist', '--calc_dist', action='store_true',
    help = "Calculate the distortion for a camera")

    parser.add_argument('-dist_img', '--dist_save_img', action='store_true',
    help= "Save generated distortion images to output folder")

    parser.add_argument('-dist_data', '--dist_save_data', action='store_true',
    help= "Save generated distortion data to output folder")

    parser.add_argument('-dist_columns', '--dist_chessboard_columns', type=int, default=9,
    help= "Number of inner column corners on calibration chessboard")

    parser.add_argument('-dist_rows', '--dist_chessboard_rows', type=int, default=7,
    help= "Number of inner row corners on calibration chessboard")

    parser.add_argument('-dist_tile_size', '--dist_chessboard_tile_size', type=float, default=0.07,
    help= "Chessboard tile size in meters")

    parser.add_argument('-d_mod', '--dist_new_dist_coeff', nargs="*", type=float, default = [],
    help= "To calculate rpe with modified distortion coefficients")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()
