
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
import requests
import os
import base64

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

@app.route('/')
def index():
    return render_template('index.html', caption='', description='')

@app.route('/display_image/<filename>')
def display_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files or 'music' not in request.files or 'description' not in request.form:
        return redirect(request.url)

    image_file = request.files['file']
    music_file = request.files['music']
    description = request.form['description']

    if image_file.filename == '' or music_file.filename == '' or description == '':
        return redirect(request.url)

    # Process the image file
    image_filename = secure_filename(image_file.filename)
    image_file.save(os.path.join(app.config['UPLOAD_FOLDER'], image_filename))

    # Process the music file
    music_filename = secure_filename(music_file.filename)
    music_file.save(os.path.join(app.config['UPLOAD_FOLDER'], music_filename))

    # Send the files to the Colab notebook's API endpoint
    ngrok_url = 'https://f03c-34-168-255-220.ngrok.io/caption'  # Replace with your ngrok URL
    files = {
        'file': open(os.path.join(app.config['UPLOAD_FOLDER'], image_filename), 'rb'),
        'music': open(os.path.join(app.config['UPLOAD_FOLDER'], music_filename), 'rb')
    }
    data = {'description': description} 
    try:
        response = requests.post(ngrok_url, files=files, data=data)

        if response.ok and response.headers.get('content-type') == 'application/json':
            api_response = response.json()
            caption = api_response.get('caption', 'Plz wait')
            img_str = api_response.get('image', '')
            filename = 'generated_image.png'

            with open(os.path.join(app.config['UPLOAD_FOLDER'], filename), 'wb') as f:
                f.write(base64.b64decode(img_str))

            return render_template('index.html', caption=caption, description=description, image=filename)
        else:
            caption = 'Unexpected response from the server'
            return render_template('index.html', caption=caption, description='', image='')

    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        return render_template('index.html', caption='Unexpected response from the server', description='', image='')


if __name__ == '__main__':
    app.run(debug=True)
