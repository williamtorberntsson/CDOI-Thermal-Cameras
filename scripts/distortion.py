"""This scripts calculates the distortion for a camera setup with the help a set of images.
The module contains the following functions:

- `CalculateDistortion` - Calculates the distortion with help of other functions.
- `find_img_mdl_points` - Calculates chessboard corner points.
- `setup_calibrate` - Calculates the camera parameters with the help of a set of points.
- `calculate_rpe_folder` -  Calculates re-projection error for all images in a folder.

Those three main functions uses other help functions:

- `find_points` - Detects chessboard corners.
- `get_calibration_object_points` - Generate a chessboard model.
- `calibrate_opencv` - Calculates camera distortion properties.
- `find_folders_with_prefix` - Find folders inside a another folder with a certain prefix.
- `find_img_in_folder` - Find all images inside a folder
"""

from pathlib import Path
import os
import sys
import csv
import cv2
import numpy as np


def CalculateDistortion(parent_folder:str, output_folder:str, camera_name:str, save_images:bool = False,
    save_data:bool = False, columns:int=9, rows:int=7, tile_size:float=0.07, d_mod:bool = None):
    """ Calculates calc_camera_parameters() with all folders starting with "Distorion" inside the given folder.
    
    Args:
        parent_folder: folder were all input folders are stored, if d_mod is given: is instead path to specific folder
        output_folder: folder where all temporary files will be stored
        camera_name: name of camera/measurement
        save_images: whereas to save images to output_folder
        save_data: whereas to save images to output_folder
        columns: Number of column (inner) corners on chessboard
        rows: Number of row (inner) corners on chessboard
        tile_size: Chessboard tile size in meters
        d_mod: Modified distortion coefficients, if specified other calculations ignored
    """

    if(d_mod):
        d_mod = np.asmatrix(d_mod) # convert to numpy array
        calculate_rpe_folder(parent_folder, output_folder, camera_name, columns, rows, tile_size, d_mod)
        return

    folders = find_folders_with_prefix(parent_folder, 'Distortion')

    if (len(folders) == 0):
        raise Exception("{} : No folders found with matching prefix and camera name!".format(parent_folder))
    else:
        print("Found {} folders to calculate distortion for.".format(len(folders)))

    for count, folder in enumerate(folders):
        print("Calculating distortion for folder: {} ({} of {} folders)\n".format(folder, count+1,len(folders)))
        # Get lens and distance from data folder name
        names = str(folder).split("/")[-1].split("_")
        lens_name = names[1]
        distance = names[2]

        mdl_points, img_points, im_size = find_img_mdl_points(parent_folder + "/" + folder, Path(output_folder / folder), save_images, save_data, columns, rows, tile_size)

        setup_calibrate(mdl_points, img_points, im_size, camera_name, lens_name, distance, output_folder)
    
    print("Distortion calculating done!!")


def find_img_mdl_points(input_folder: str, output_folder: str, save_images:bool, save_data:bool,
    columns:int, rows:int, tile_size:float) -> tuple[list, list, tuple[int,int]]:
    """ Setup for detecting chessboard corners and calculating chessboard points.

    Args:
        input_folder: Absolute path for folder with all input images
        output_folder: Absolute path for folder where all output data should be saved
        save_images: If true, new images with detected points drawn will be generated
        save_data: Data points will be saved to files in output folder
        columns: Number of column (inner) corners on chessboard
        rows: Number of row (inner) corners on chessboard
        tile_size: Chessboard tile size in meters
    
    Returns:
        results: A list constisting of a dataset from every image
    """

    # Stores name of images that were successful, unsuccessful and not readable 
    images_result = {"successful": [], "unsuccessful": [], "unread": []}

    # Input directory to read images from
    images_path = Path(input_folder).expanduser()
    print("Images path: {}".format(images_path))

    # Output directory for saving visualizations and results
    if(save_data or save_images):
        data_path = Path(output_folder).expanduser()
        data_path.mkdir(parents=True, exist_ok=True)
        print("Data will be saved in: {}".format(output_folder))

    # Chessboard inner corners, (columns, rows)
    cb_inner_corners = (columns, rows)

    mdl_pts = get_calibration_object_points(cb_inner_corners, tile_size)

    results = []

    im_size = None

    files = find_img_in_folder(images_path)

    # List to store points
    img_points = []
    mdl_points = []

    for i, fname in enumerate(files):
        sys.stdout.write("\033[K") # clear previus line
        print("\tDetecting corners in {} ({} of {})".format(fname.name, i+1, len(files)), end="\r")

        # Load image
        im = cv2.imread(str(fname))
        if im is None:
            images_result["unread"].append(fname.name)
            continue
        if im_size is None:
            im_size = im.shape[:2]
        else:
            try:
                assert im.shape[:2] == im_size
            except AssertionError:
                raise Exception("Images are not the same size!")

        # Find corners
        found, im_pts = find_points(im, cb_inner_corners)
        if not found:
            images_result["unsuccessful"].append(fname.name)
            continue
        
        # Add points to list
        img_points.append(im_pts)
        mdl_points.append(mdl_pts)
        images_result["successful"].append(fname.name)

        # Visualize and save
        if(save_images):
            vis = cv2.drawChessboardCorners(im, cb_inner_corners, im_pts, found)
            cv2.imwrite(str(data_path / (fname.stem + ".png")), vis)
        
        data = dict(cb_inner_corners=cb_inner_corners, tilesize=tile_size,
                        im_size=im_size[::-1], image_points=im_pts, model_points=mdl_pts)
        results.append(data)
        if(save_data):
            np.savez_compressed(data_path / (fname.stem + ".npz"), **data)

    # Prints to terminal
    if(len(images_result["unsuccessful"]) == 0 and len(images_result["successful"]) > 0 and len(images_result["unread"]) == 0):
        #sys.stdout.write("\033[K") # clear previus line
        print("\nFound all corners in all images!")
    else:
        #sys.stdout.write("\033[K") # clear previus line
        print("Could not find all corners in {} out of {} images! Data for all other images were generated!".format(len(images_result["unsuccessful"]), len(images_result["unsuccessful"]) + len(images_result["successful"])))
        if(len(images_result["unsuccessful"]) > 0):
            print("Could not find all corners in following images!")
            for name in images_result["unsuccessful"]:
                print("\t{}".format(name))
        if(len(images_result["unread"]) > 0):
            print("Could not read following images!")
            for name in images_result["unread"]:
                print("\t{}".format(name))
    
    return mdl_points, img_points, im_size


def setup_calibrate(model_points:list, image_points:list, im_size:tuple[int,int], camera_name:str, lens_name:str, distance:str, output_path:Path):
    """ Setup for calibrating camera with OpenCV's method.
    
    Args:
        model_points: np.array of 3D model points, shape (N, 3) (N columns, 3 rows)
            In each column, the coordinates are ordered x,y,z
        image_points: np.array of 2D image pixel coordinates, shape (N, 2) in the same order
            as the model points.
        im_size (tuple): tuple (width, height) in pixels
        camera_name: Name of camera, used when saving output
        distance: Distance used in measurement, used when saving output
        output_path: Path to where output will be saved
    """

    A, d, Rs, ts, rpe = calibrate_opencv(model_points, image_points, im_size)

    # Save to file
    new_row = '-' * 25
    with open(output_path / 'output.csv', 'a', encoding='UTF8') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_NONE, escapechar=' ')
        writer.writerow(["{} Camera {} with lens {} at a distance of {} {}".format(new_row, camera_name, lens_name, distance, new_row)])

        writer.writerow(["Camera intrinsics:"])
        writer.writerow([",{:.2f},0,{:.2f}".format(A[(0,0)],A[(0,2)])])
        writer.writerow([",0,{:.2f},{:.2f}".format(A[(1,1)],A[(1,2)])])
        writer.writerow([",0,0,1"])
        writer.writerow(' ')

        writer.writerow(["Distortion coefficients:,k1,k2,p1,p2,k3"])
        writer.writerow([",{:.3f},{:.3f},{:.3f},{:.3f},{:.3f}".format(d[0,0],d[0,1],d[0,2],d[0,3],d[0,4])])
        writer.writerow(' ')

        writer.writerow(["Mean reprojection error:,{:.3f} pixels".format(rpe)])
        writer.writerow(' ')

    print("Calculations saved to {}\n".format(output_path / 'output.csv'))

    #Save the calibration for later use

    #cal_data = dict(A=A.tolist(), d=d.tolist(), im_size=tuple(map(int, im_size)),
    #                Rs=np.stack(Rs).tolist(), ts=np.stack(ts).tolist(), rpe=rpe)
    #json.dump(cal_data, open(output_path / "calibration_cv.json", "w"))


def calculate_rpe_folder(input_folder:str, output_folder:str, camera_name:str, columns:str, rows:int, tile_size:int, d_mod:list[float]):
    """ Calculates mean re-projection error for all images in a folder.
    
    Args:
        input_folder: Absolute path for folder with all input images
        output_folder: Absolute path for folder where all output data should be saved
        columns: Number of column (inner) corners on chessboard
        rows: Number of row (inner) corners on chessboard
        tile_size: Chessboard tile size in meters
        d_mod: Modified distortion coefficients
    """

    print("\nCalculating rpe with modified distortion for folder: {}\n".format(input_folder))
    # Get lens and distance from data folder name
    if(input_folder[-1] == '/'):
        names = str(input_folder).split("/")[-2].split("_")
    else:
        names = str(input_folder).split("/")[-1].split("_")
    lens_name = names[1]
    distance = names[2]

    mdl_points, img_points, im_size = find_img_mdl_points(input_folder, output_folder, False, False, columns, rows, tile_size)

    A, d, Rs, ts, rpe = calibrate_opencv(mdl_points, img_points, im_size)

    mod_mean_error = 0
    mean_error = 0
    # Calculate mean re-projection error over all images
    # modified distortion coefficients
    for i in range(len(mdl_points)):
            imgpoints2, _ = cv2.projectPoints(mdl_points[i], Rs[i], ts[i], A, d_mod)
            error = cv2.norm(img_points[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
            mod_mean_error += error
    
    # original distortion coefficients
    for i in range(len(mdl_points)):
            imgpoints2, _ = cv2.projectPoints(mdl_points[i], Rs[i], ts[i], A, d)
            error = cv2.norm(img_points[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
            mean_error += error
    
    # Normalize error
    mod_mean_error = mod_mean_error / len(img_points)
    mean_error = mean_error / len(img_points)
    
    # Write results to output file
    new_row = '-' * 25
    with open(output_folder / 'output.csv', 'a', encoding='UTF8') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_NONE, escapechar=' ')
        writer.writerow(["{} Camera {} with lens {} at a distance of {} {}".format(new_row, camera_name, lens_name, distance, new_row)])

        writer.writerow(["Modified distortion coefficients:,k1,k2,p1,p2,k3"])
        writer.writerow([",{:.3f},{:.3f},{:.3f},{:.3f},{:.3f}".format(d_mod[0,0],d_mod[0,1],d_mod[0,2],d_mod[0,3],d_mod[0,4])])
        writer.writerow([" "])
        writer.writerow(["Original distortion coefficients:,k1,k2,p1,p2,k3"])
        writer.writerow([",{:.3f},{:.3f},{:.3f},{:.3f},{:.3f}".format(d[0,0],d[0,1],d[0,2],d[0,3],d[0,4])])
        writer.writerow([" "])

        writer.writerow(["Mean reprojection error,{:.3f} pixels".format(mean_error)])
        writer.writerow(["Modified mean reprojection error,{:.3f} pixels".format(mod_mean_error)])
        writer.writerow(["CalibrateCamera reprojection error,{:.3f} pixels".format(rpe)])
        writer.writerow([" "])

    print("Results saved to {}".format(output_folder / 'output.csv'))


##### Help functions #####

def find_points(rgb_image:cv2.Mat, inner_points:tuple[int,int]) -> tuple[bool,np.ndarray|None]:
    """ Detect chessboard corners in an image.

    Args:
        rgb_image:  input image
        inner_points: (w, h) - chessboard inner corners, columns and rows
    
    Returns:
        found: if all points where found
        points: points is an (h*w,2) array of (x,y) image pixel coordinates
    """
    gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
    found, points = cv2.findChessboardCorners(gray, inner_points)
    if found:
        term_crit = cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001
        points = cv2.cornerSubPix(gray, points, (11, 11), (-1, -1), term_crit)
        return found, points
    return False, None


def get_calibration_object_points(inner_points:tuple[int,int], tile_size:float) -> np.ndarray:
    """ Generate chessboard points.

    Args:
        inner_points: (w, h) - chessboard inner corners, columns and rows
        tile_size: chessboard tile size [meter]
    
    Returns:
        obj_points: Array of chessboard points shape (h*w, 3), ordered left-to-right, top-to-bottom. This ordering must match the output of find_points()
    """
    w, h = inner_points
    obj_points = np.empty((h*w, 3), np.float32)

    i = 0
    for y in range(h):
        for x in range(w):
            obj_points[i] = (x * tile_size, y * tile_size, 0.0)
            i += 1

    return obj_points


def calibrate_opencv(model_points:list, image_points:list, im_size:tuple[int,int]) -> tuple[np.ndarray,np.ndarray,list,tuple,float]:
    """ Calibrate with OpenCV's method
    
    Args:
        model_points: np.array of 3D model points, shape (N, 3) (N columns, 3 rows)
            In each column, the coordinates are ordered x,y,z
        image_points: np.array of 2D image pixel coordinates, shape (N, 2) in the same order
            as the model points.
        im_size (tuple): tuple (width, height) in pixels
    
    Returns:
        A: Tuple of camera intrinsics
        d: Distortion coefficients
        Rs: Camera rotation matrices
        ts: Camera translation vectors
        rpe: Average reprojection error [pixels]
    """

    flags = 0
    # Enable these to disable all but the distortion coefficients used in Zhang's method:
    # flags = cv2.CALIB_ZERO_TANGENT_DIST + cv2.CALIB_FIX_K3

    rpe, A, d, rs, ts = cv2.calibrateCamera(model_points, image_points, im_size, None, None, flags=flags)
    Rs = [cv2.Rodrigues(r)[0] for r in rs]  # Convert Rodrigues rotation to rotation matrices

    return A, d, Rs, ts, rpe


def find_folders_with_prefix(parent_folder:str, prefix:str) -> list[str]:
    """ Find all folders inside a folder with a certain prefix.

    Args:
        parent_folder: Folder to search for other folders inside
        prefix: Prefix to search for

    Returns:
        folder_list: list of folder names that matched the prefix
    """ 
    parent_folder = Path(parent_folder).resolve()

    folder_list = []
    for folder in os.listdir(parent_folder):
        name_folder = folder.split("_")
        if name_folder[0] == prefix:
            folder_list.append(folder)

    return folder_list

def find_img_in_folder(folder:str) -> list[str]:
    """ Finds all img files inside a folder.

    Args:
        folder: Path to folder
    
    Returns:
        files: List will paths to found filenames
    """
    # Source image file name pattern
    file_patterns = ["*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg"]

    files = []
    for pattern in file_patterns:
        files += list(sorted(folder.glob(pattern)))
    
    return files
