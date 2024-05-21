#!/bin/bash
# Place and run this in the directory where you want the files to be downloaded (for m=4 use the folder R1M1BH10000beta1, form m=5 R0.47M0.5BH100000beta1S60n1.5Compton)

# Download
for d in 196 # Change the range to the one you want
do
    rclone copy drive:/Snellius/R1M1BH100000beta2S60n3Compton/snap_full_"$d".h5 . -P 
                                                                     
done                                                                 

# Make orderly
for d in 196 # Again, these are snapshot numbers
do
    mkdir "$d"
    mv snap_"$d".h5 "$d"/snap_full_"$d".h5
done

echo "Job's done!"

 