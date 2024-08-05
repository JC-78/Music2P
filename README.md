
# Music2P
Multi-modal Music ML service that will fulfil promotion needs of musicians and companies. All you need to do is provide music, image and write what kind of album promotion cover you want.

# Generate Captions
There are two strategies to generate captions. One is 'Music-to-Caption' via BART. And the other is 'Image-to-Caption' via BLIP.
* ```custom_data``` has mp3 files and jpg files.
* ```demo/app.py``` gets the mp3 file and outputs caption by BART. It also generates captions based on an image by BLIP.

# How to use Music2P
Run ColabToFlask.ipynb colab notebook, get the url generated and paste that into app.py's url variable and run app.py. It will run Music2P on your local host.
