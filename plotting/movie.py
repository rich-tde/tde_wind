abspath = '/Users/paolamartire/shocks'
import subprocess

# Choose simulation
m = 4
Mbh = 10**m
beta = 1
mstar = .5
Rstar = .47
n = 1.5
compton = 'Compton'
check = 'Low'
cut = '' # or '' or 'cut' or 'lowcut'
npanels = 3
folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}'

# For Denproj
# path = f'{abspath}/Figs/{folder}/{check}/projection/denproj'
# output_path = f'{abspath}/Figs/{folder}/movie_proj{check}{cut}.mp4'

# For Slices 
path = f'{abspath}/Figs/{folder}/{check}/slices/Panel{npanels}Slice'
output_path = f'{abspath}/Figs/{folder}/{check}/movie{npanels}Panels_{cut}.mp4'

start = 100
slow_down_factor = 2  # Increase this value to make the video slower

# ffmpeg_command = (
#     f'ffmpeg -y -start_number {start} -i {path}%03d{cut}.png -vf "setpts={slow_down_factor}*PTS" '
#     f'-frames:v {end_frame+ 1} -c:v libx264 -pix_fmt yuv420p {output_path}'
# )

ffmpeg_command = (
    f'ffmpeg -y -start_number {start} -i {path}%d{cut}.png -vf "setpts={slow_down_factor}*PTS" '
    f'-c:v libx264 -pix_fmt yuv420p {output_path}'
    )

subprocess.run(ffmpeg_command, shell=True)

