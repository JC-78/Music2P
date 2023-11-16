
# Prerequisites to run the code
These libraries are essential for BLIP.
* transformers==4.15.0
* timm==0.4.12
* fairscale==0.4.4

After that, clone the BLIP repository.
* git clone https://github.com/salesforce/BLIP



# Music2P
Multi-modal Music ML service that will fulfil promotion needs of musicians and companies 

# Generate Captions
There are two strategies to generate captions. One is 'Music-to-Caption' via BART. And the other is 'Image-to-Caption' via BLIP.
* ```custom_data``` has mp3 files and jpg files.
* ```demo/app.py``` gets the mp3 file and outputs caption by BART. It also generates captions based on an image by BLIP.

