from keras.applications.vgg19 import VGG19 
from keras.models import sequential
from keras.layers import Dense, Dropout, Flatten

from keras.preprocessing.image import ImageDataGenerator

# Download current VGG19 model, withoutht the last layer
# and all layers are NOT trainable
vgg19 = VGG19(include_top=False, # don't get the final dense classification layer
              weights='imagenet', # get the trained parameters 
              input_shape=(244,244,3), # the data we will use
              pooling=None) #

for layer in vgg19.layers:
    layer.trainable = False

# Add the layers for our task at hand
model = Sequential()
model.add(vgg19)

model.add(Flatten(name='flattened'))
model.add(Dropout(0.5, name='dropout'))
model.add(Dense(2, activation='softmax', name='predictions'))

model.compile(optimizer='adam', loss='categorical_crossentropy',
              metris=['accuracy'])

# Two image generator classes
train_dategen = ImageDataGenerator(
    rescale=1.0/255,
    data_format='channels_last',
    rotation_range=30,
    horizontal_flip=True,
    fill_mode='reflect'
)

valid_datagen = ImageDataGenerator(
    rescale=1.0/255,
    data_format='channels_last'
)

batch_size=32

train_generator = train_datagen.flow_from_dictionary(
    directory='./hot-dog-not-hot-dog/train',
    target_size=(224,244),
    classes=['hot-dog', 'not-hot-dog'],
    batch_size=batch_size,
    shuffle=True,
    seed=42)

valid_generator = train_valid.flow_from_dictionary(
    directory='./hot-dog-not-hot-dog/test',
    target_size=(224,244),
    classes=['hot-dog', 'not-hot-dog'],
    batch_size=batch_size,
    shuffle=True,
    seed=42)

model.fit_generator(train_generator, steps_per_epoch=15,
                    epochs=16, validation_data=valid_generator,
                    validation_steps=15)



                    