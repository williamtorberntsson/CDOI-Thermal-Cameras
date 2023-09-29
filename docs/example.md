# Example calls
Here are a few examples of different calls that can be used to calculate different parameters.
The output is stored to the root folder of the project.

## Calculate all metrics for a camera
Enter an input directory to where suitable data for each metric is stored, and use the -ALL_METRICS flag as in 
```bash
python3 scripts/main.py -i ~/[folder-with-images]/[camera-name] -a
```
which e.g. could be 
```bash
python3 scripts/main.py -i ~/TSBB11/cdio_thermal_cameras/A50 -a
```
for the uncooled camera A50. This would produce values for temperature standard deviation and NEDT in a csv-file as well as MTF png-plots, further stored in
```bash
~/TSBB11/cdio_thermal_cameras/output/A50
```
since no distortion measurements are stored in the input directory.

## Calculate distortion for a camera
```bash
python3 scripts/main.py -i ~/[folder-with-images]/[camera-name] -dist
```

## Calculate distortion with different distortion coefficients
```bash
python3 scripts/main.py -i ~/[folder-with-images]/[camera-name]/[measurement] -dist -d_mod k1 k2 p1 p2 k3
```
for example:
```bash
python3 scripts/main.py -i ~/Measurements/X8400/Distortion_50mm_212cm -dist -d_mod 0.5 0.3 0.1 2.1 3.2
```