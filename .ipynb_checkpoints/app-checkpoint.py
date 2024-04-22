from flask import Flask, jsonify, request, render_template
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import io
import base64
import glob
import threading
import pyart
from flask_frozen import Freezer

app = Flask(__name__)
freezer = Freezer(app)

inputs = np.sort(glob.glob('/Users/piyushg/Desktop/goes_mrms/inputs_clean*.npy'))
outputs = np.sort(glob.glob('/Users/piyushg/Desktop/goes_mrms/outputs_clean*.npy'))

input_array = np.load(inputs[0])
output_array = np.load(outputs[0])

n_times = input_array.shape[0]

# Create a threading lock
lock = threading.Lock()

# Dictionary to map channel numbers to names
channel_names = {
    0: 'Channel 8',
    1: 'Channel 9',
    2: 'Channel 10',
    3: 'Channel 13'
}

def display_images(timestep, channel):
    with lock:
        fig = plt.figure(figsize=(14, 5))

        proj = ccrs.PlateCarree()
        extent = [-125, -40, 24, 50]

        ax1 = fig.add_subplot(1, 2, 1, projection=proj)
        im1 = ax1.imshow(input_array[timestep, channel], cmap='turbo', extent=extent, transform=proj)
        ax1.set_title(f"GOES Radiances - Timestep: {timestep}, Channel: {channel_names[channel]}")
        ax1.coastlines()
        g1 = ax1.gridlines(draw_labels=True)
        g1.top_labels = g1.right_labels = False

        cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.09, orientation='horizontal')
        cbar1.ax.tick_params(labelsize=8)

        ax2 = fig.add_subplot(1, 2, 2, projection=proj)
        im2 = ax2.imshow(output_array[timestep], cmap='pyart_LangRainbow12', extent=extent, transform=proj)
        ax2.set_title(f"MRMS Composite Ref - Timestep: {timestep}")
        ax2.coastlines()
        g2 = ax2.gridlines(draw_labels=True)
        g2.top_labels = g2.right_labels = False

        cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.09, orientation='horizontal')
        cbar2.ax.tick_params(labelsize=8)

        plt.tight_layout()

        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png')
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
        plt.close(fig)

    return img_base64

@app.route('/update_plot')
def update_plot():
    timestep = request.args.get('timestep')
    channel = request.args.get('channel')
    
    if timestep is None or channel is None:
        # Return a default plot image if timestep or channel is missing
        default_timestep = 0
        default_channel = 0
        plot_data = display_images(default_timestep, default_channel)
        return jsonify({'plot': plot_data})
    
    try:
        timestep = int(timestep)
        channel = int(channel)
    except ValueError:
        # Return a default plot image if timestep or channel is invalid
        default_timestep = 0
        default_channel = 0
        plot_data = display_images(default_timestep, default_channel)
        return jsonify({'plot': plot_data})
    
    plot_data = display_images(timestep, channel)
    return jsonify({'plot': plot_data})

@app.route('/')
def index():
    return render_template('index.html', n_times=n_times, channel_names=channel_names)

@freezer.register_generator
def update_plot_generator():
    for timestep in range(n_times):
        for channel in range(4):
            yield 'update_plot', {'timestep': timestep, 'channel': channel}
    
if __name__ == '__main__':
    freezer.freeze()
    

