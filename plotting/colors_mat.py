from matplotlib.gridspec import GridSpec
from matplotlib import colors as mcolors
from matplotlib import pyplot as plt

colors = mcolors._colors_full_map #dictionary of all colors

#sort them by hsv
by_hsv = sorted((tuple(mcolors.rgb_to_hsv(mcolors.to_rgba(color)[:3])), name)
                for name, color in colors.items())
sorted_names = [name for hsv, name in by_hsv]

def plot_all_colors():
    fig = plt.figure(figsize=(17,35),dpi=200)
    grid = GridSpec(195,6,hspace=1,wspace=2)
    counter = 0
    while counter < 1163:
        column = (counter//195)%6
        row = counter%195
        ax = fig.add_subplot(grid[row,column])
        ax.axis('off')
        ax.axhline(0,linewidth=15,c=sorted_names[counter])
        color_str = sorted_names[counter]
        #recall that xkcd colors must be prefaced with "xkdc:". To save space, I'll take that out
        #and replace with an asterisk so we can still identify them
        # if 'xkcd' in color_str:
        #     color_str = color_str[5:]+'*'
        ax.text(-0.03,0.5,color_str,ha='right',va='center')
        counter+=1
    else:
        ax = fig.add_subplot(grid[-4,-1])
        ax.axis('off')
        ax.text(0,0,'* = xkcd')
#         fig.savefig('matplotlib named colors.png', bbox_inches='tight') #uncomment to save figure
            
plot_all_colors()