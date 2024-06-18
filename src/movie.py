import sys
sys.path.append('/Users/paolamartire/tde_comparison')

import subprocess
import glob
import os

# Choose simulation
m = 4
Mbh = 10**m
beta = 1
mstar = .5
Rstar = .47
n = 1.5
folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}'
path = f'Figs/{folder}/multiple/wH'
output_path = f'Figs/{folder}/movie_wH.mp4'

start = 1
slow_down_factor = 6  # Increase this value to make the video slower

ffmpeg_command = (
    f'ffmpeg -y -start_number {start} -i {path}%d.png -vf "setpts={slow_down_factor}*PTS" '
    f'-c:v libx264 -pix_fmt yuv420p {output_path}'
)

subprocess.run(ffmpeg_command, shell=True)

