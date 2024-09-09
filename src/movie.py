abspath = '/Users/paolamartire/shocks'
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
compton = 'Compton'
check = 'Low'
folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}'

# For Denproj
# path = f'{abspath}/Figs/{folder}/{check}/projection/denproj'
# output_path = f'{abspath}/Figs/{folder}/movie_proj{check}{cut}.mp4'

# For Slices 
path = f'{abspath}/Figs/{folder}/{check}/slices/midplaneRad_'
output_path = f'{abspath}/Figs/{folder}/{check}/movie_Rad.mp4'

start = 100
slow_down_factor = 4  # Increase this value to make the video slower

ffmpeg_command = (
    f'ffmpeg -y -start_number {start} -i {path}%d.png -vf "setpts={slow_down_factor}*PTS" '
    f'-c:v libx264 -pix_fmt yuv420p {output_path}'
)

subprocess.run(ffmpeg_command, shell=True)

