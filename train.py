from model import *

#made up 3D image data for segmentation 
label = numpy.random.randint(2,size=(10,64,64,64,1))
training_data = numpy.random.randint(256,size=(10,64,64,64,1))

model = unet()

model.fit(x=training_data, y=label, epochs=10)
