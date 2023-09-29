"""This file contains a script to calculate the temperature differens from measured with a cooled camera with an uncooled camera

The module contains the following functions:

- `CalculateStdTemperature` - Returns the standard deviation between a uncooled camera and a cooled camera.

Examples:
    >>> from scripts import temperature
    >>> temperature.CalculateStdTemperature('.\Measurements\MeasuredTemperatures.txt','.\Measurements\A50')
    'Standard deviation is ±1.2°C'
    >>> temperature.CalculateStdTemperature('.\Measurements\MeasuredTe.txt')
    "Path is not Valid"
"""

import numpy as np

def CalculateStdTemperature(filename:str, camera_name:str) -> tuple[list,list]:
    """Compute and return the standard-deviation of measured vs actual temperature.

    Args:
        filename: Textfile with the measured temperatures from the cooled and uncooled camera 
        camera_name: Takes in the camera-name from a folder from the main-script

    Returns:
        header (list): A list with a text with the cameraname along.
        value (list): A list with the resulting standarddeviation of the temperature.
    """    

    print("Running temperature calculations")

    header = []
    value = []
    try:
        # Try to read and open the filename
        lines = open(filename).read().split('\n')
    except:
        lines = []
        # If the file can not be opened the function returns this.
        return "Path is not valid", None

    if lines != []: # If path is valid

        for line in lines:
            line = line.split(' ')
        std_list = np.zeros(((len(lines)-1),len(line)-1)) # Create a list to save values in
        camera_names = []

        line_nbr = 0
        for line in lines[1:]: # Read all lines in list
            line = line.split(' ')
            temperature_nbr = 0
            for temps in line:
                try:
                    # See if item in list can be turned to float
                    temps = float(temps)
                    # If it is a number att to the list
                    std_list[line_nbr][temperature_nbr] = temps
                except:
                    camera_names.append(temps.rstrip(':'))
                    continue
                temperature_nbr += 1
            line_nbr += 1 
        # Calculate the standard deviation from a sample of data
        std_value = []
        row = 0
        for temp in std_list[1:,:]:
            arr = np.vstack((temp, std_list[0,:]))
            # Calculate the std from the array with the temperatures
            std_value = np.std(arr, axis=0)
            # Calculate the mean-std for the array
            mean_std = np.mean(std_value)
            if camera_names[row + 1] == camera_name:
                header.append(f"Standard deviation for {camera_name} temperature measurements compared against cooled camera:")
                value.append(f" ± {str(round(mean_std,2))} °C")
            row += 1
        if header == []:
            print(f"No vaild temperature measurements for {camera_name} in input file. Continuing with calculations")
        return header, value
