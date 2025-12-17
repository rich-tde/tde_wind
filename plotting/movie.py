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
check = 'HiResNewAMR'
folder = f'R{Rstar}M{mstar}BH{Mbh}beta{beta}S60n{n}{compton}{check}'

start = 21
slow_down_factor = 3 # Increase this value to make the video slower

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
how_many = ''
path = f'{abspath}/Figs/{folder}/projection{how_many}/denproj_%d.png' 
output_path = f'{abspath}/Figs/{folder}/projection{how_many}/denproj_{how_many}{check}.mp4'
first_image_path = f'{abspath}/Figs/{folder}/projection{how_many}/denproj_{start}.png'

# For Slices 
# path = f'{abspath}/Figs/{folder}/slices/Panel6Slice%d.png' 
# output_path = f'{abspath}/Figs/{folder}/slices/movie6Panels_{check}.mp4'
# first_image_path = f'{abspath}/Figs/{folder}/slices/Panel6Slice%d{start}.png'

# For Opacity
# path = f'{abspath}/Figs/{folder}/testOpac/{check}_TestOpac%d.png' 
# output_path = f'{abspath}/Figs/{folder}/testOpac/TestOpac_movie.mp4'
# first_image_path = f'{abspath}/Figs/{folder}/testOpac/{check}_TestOpac{start}.png'

# For outflow
# path = f'{abspath}/Figs/{folder}/Outflow/B_slice_%d.png' 
# output_path = f'{abspath}/Figs/{folder}/Outflow/B_{check}.mp4'
# first_image_path = f'{abspath}/Figs/{folder}/Outflow/B_slice_{start}.png'

# For stream
# path = f'{abspath}/Figs/{folder}/stream/WH_theta%d.png'
# output_path = f'{abspath}/Figs/{folder}/stream/WH_theta.mp4'
# first_image_path = f'{abspath}/Figs/{folder}/stream/WH_theta{start}.png'


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