import os
from PIL import Image

for subdir, dirs, files in os.walk('./'):
    for file in files:
        if(file.endswith('.png')):
            img = Image.open(file)
            li = file.split('.')[0]
            num = li[2:]
            img.save('tar'+num+'.png')
