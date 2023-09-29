## How to run the program

Use help flag on main.py to see all required and optional inputs for calculating desired metrics of a camera.
```bash
python3 scripts/main.py -h

usage: main.py [-h] -i INPUT_IMAGE_DIR [-o OUTPUT_DIR] [-rmo]
               [-it INPUT_TEMP_DIR] [-std] [-nedt]
               [-crop CROPPED_IMAGE_FACTOR] [-mtf] [-a] [-dist]
               [-dist_img] [-dist_data]
               [-dist_columns DIST_CHESSBOARD_COLUMNS]
               [-dist_rows DIST_CHESSBOARD_ROWS]
               [-dist_tile_size DIST_CHESSBOARD_TILE_SIZE]
               [-d_mod [DIST_NEW_DIST_COEFF [DIST_NEW_DIST_COEFF ...]]]

Computes metrics for input sequence of thermal images

optional arguments:
  -h, --help            show this help message and exit
  -o OUTPUT_DIR, --output_dir OUTPUT_DIR
                        Output directory for csv-file with derived
                        parameters (default: output)
  -rmo, --remove_output_dir
                        Clear output directory for csv-file with derived
                        parameters before creating new (default: False)
  -it INPUT_TEMP_DIR, --input_temp_dir INPUT_TEMP_DIR
                        Input directory to .txt file containing
                        temperature measurements (default: None)
  -std, --calc_std_temp
                        Calulate standard deviation for temperature
                        measurements in .txt file (default: False)
  -nedt, --calc_nedt    Calculate NEDT-value for a camera with a specific
                        integration time in a folder (default: False)
  -crop CROPPED_IMAGE_FACTOR, --cropped_image_factor CROPPED_IMAGE_FACTOR
                        Desired factor to crop images for calculating NEDT
                        (default: 0.8)
  -mtf, --calc_mtf      Calculate the MTF for a camera (default: False)
  -a, --all_metrics     Calculate all metrics for a camera (default:
                        False)
  -dist, --calc_dist    Calculate the distortion for a camera (default:
                        False)
  -dist_img, --dist_save_img
                        Save generated distortion images to output folder
                        (default: False)
  -dist_data, --dist_save_data
                        Save generated distortion data to output folder
                        (default: False)
  -dist_columns DIST_CHESSBOARD_COLUMNS, --dist_chessboard_columns DIST_CHESSBOARD_COLUMNS
                        Number of inner column corners on calibration
                        chessboard (default: 9)
  -dist_rows DIST_CHESSBOARD_ROWS, --dist_chessboard_rows DIST_CHESSBOARD_ROWS
                        Number of inner row corners on calibration
                        chessboard (default: 7)
  -dist_tile_size DIST_CHESSBOARD_TILE_SIZE, --dist_chessboard_tile_size DIST_CHESSBOARD_TILE_SIZE
                        Chessboard tile size in meters (default: 0.07)
  -d_mod [DIST_NEW_DIST_COEFF [DIST_NEW_DIST_COEFF ...]], --dist_new_dist_coeff [DIST_NEW_DIST_COEFF [DIST_NEW_DIST_COEFF ...]]
                        To calculate rpe with modified distortion
                        coefficients (default: [])

required:
  -i INPUT_IMAGE_DIR, --input_image_dir INPUT_IMAGE_DIR
                        Input directory to folder file containing .tif-
                        files (default: None)

```