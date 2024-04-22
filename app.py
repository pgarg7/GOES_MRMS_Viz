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
app = Flask(__name__)

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

@app.route('/update_plot', methods=['GET'])
def update_plot():
    timestep = int(request.args.get('timestep'))
    channel = int(request.args.get('channel'))

    plot_data = display_images(timestep, channel)

    return jsonify({'plot': plot_data})

@app.route('/')
def index():
    return render_template('index.html', n_times=n_times, channel_names=channel_names)

if __name__ == '__main__':
    app.run(threaded=False)
    

