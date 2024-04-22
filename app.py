from flask import Flask, send_file, render_template, request
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use the 'Agg' backend for generating images without a display
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import base64
from flask_frozen import Freezer
import glob
import pyart
import io

app = Flask(__name__)
freezer = Freezer(app)

inputs = np.sort(glob.glob('/Users/piyushg/Desktop/goes_mrms/inputs_clean*.npy'))
outputs = np.sort(glob.glob('/Users/piyushg/Desktop/goes_mrms/outputs_clean*.npy'))

input_array = np.load(inputs[0])
output_array = np.load(outputs[0])

n_times = 100

channel_names = {
    0: 'Channel 8',
    1: 'Channel 9',
    2: 'Channel 10',
    3: 'Channel 13'
}

def display_images(timestep, channel):
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

# Create a directory to store the images
#os.makedirs('static/images', exist_ok=True)

# Pre-generate all the images
#for timestep in range(n_times):
#    for channel in range(4):
#        img_base64 = display_images(timestep, channel)
#        img_filename = f'static/images/timestep_{timestep}_channel_{channel}.png'
#        with open(img_filename, 'wb') as f:
#            f.write(base64.b64decode(img_base64))

@app.route('/update_plot')
def update_plot():
    timestep = 0
    channel = 0
    if request:
        timestep = request.args.get('timestep', default=0, type=int)
        channel = request.args.get('channel', default=0, type=int)
    img_filename = f'static/images/timestep_{timestep}_channel_{channel}.png'
    return send_file(img_filename, mimetype='image/png')

@app.route('/')
def index():
    return render_template('index.html', n_times=n_times, channel_names=channel_names)

@freezer.register_generator
def url_generator():
    for timestep in range(n_times):
        for channel in range(4):
            yield 'update_plot', {'timestep': timestep, 'channel': channel}

if __name__ == '__main__':
    freezer.freeze()