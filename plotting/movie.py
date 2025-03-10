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
check = ''
npanels = 6
folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'

start = 80
slow_down_factor = 2  # Increase this value to make the video slower

# Get the height of the first image to calculate the scale
def get_image_size(image_path):
    result = subprocess.run(['ffprobe', '-v', 'error', '-show_entries',
                             'stream=width,height', '-of', 'default=noprint_wrappers=1', image_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE)
    output = result.stdout.decode('utf-8')
    width = int(output.split("width=")[1].split("\n")[0])
    height = int(output.split("height=")[1].split("\n")[0])
    return width, height

# For Denproj
path = f'{abspath}/Figs/talk/denproj_diss%d.png' #{folder}/projection/denproj%d.png'
output_path = f'{abspath}/Figs/talk/denproj_diss{check}.mp4' #{folder}/movie_proj{check}.mp4'
first_image_path = f'{abspath}/Figs/talk/denproj_diss100.png' #{folder}/projection/denproj{start}.png'

# For Slices 
# path = f'{abspath}/Figs/EddingtonEnvelope/vel/vel_%d.png' 
# output_path = f'{abspath}/Figs/EddingtonEnvelope/vel/collection_vel.mp4'
# first_image_path = f'{abspath}/Figs/EddingtonEnvelope/vel/vel_{start}.png'

# For res insensitivity Rph
# path = f'{abspath}/Figs/Test/photosphere/348/348_RinRph_ray%d.png'
# output_path = f'{abspath}/Figs/Test/photosphere/348/348_RinRph_movie.mp4'
# start = 0
# first_image_path = f'{abspath}/Figs/Test/photosphere/348/348_RinRph_ray{start}.png'

width, height = get_image_size(first_image_path)

# Ensure the height is even by subtracting 1 if it's odd.
if height % 2 != 0:
    new_height = height - 1
else:
    new_height = height

# Ensure width and height are even
if width % 2 != 0:
    width -= 1
if height % 2 != 0:
    height -= 1

# Construct the FFmpeg command with scaling
ffmpeg_command = (
    f'ffmpeg -y -start_number {start} -i "{path}" -vf "setpts={slow_down_factor}*PTS,scale={width}:{new_height}" '
    f'-c:v libx264 -pix_fmt yuv420p "{output_path}"'
)

subprocess.run(ffmpeg_command, shell=True, check=True) # added check=True

print('Done')