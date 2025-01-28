from keras import applications
from keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.layers import Flatten, Dense, Dropout
from keras.callbacks import ModelCheckpoint
import numpy as np
import keras

# Define image dimensions
img_width, img_height = 48, 48


# Define directories for training and validation data
train_data_dir = r"C:\Users\Elite\Downloads\FYP project\LANDING AI PROJECT\Chars-74k dataset\EnglishFnt\English"
validation_data_dir = r"C:\Users\Elite\Downloads\FYP project\LANDING AI PROJECT\Chars-74k dataset\EnglishFnt\validation"

# Define other parameters
batch_size = 32
epochs = 20
num_classes = 62  # Update this value to match the number of classes in your dataset

# Load pre-trained VGG19 model
model = applications.VGG19(weights="imagenet", include_top=False, input_shape=(img_width, img_height, 3))

# Freeze first five layers
for layer in model.layers[:5]:
    layer.trainable = False

# Build top layers
x = Flatten()(model.output)
x = Dense(32, activation="tanh")(x)
x = Dropout(0.5)(x)
x = Dense(128, activation="tanh")(x)
predictions = Dense(num_classes, activation="softmax")(x)  # Use num_classes instead of hardcoded value

# Create final model
model_final = Model(inputs=model.input, outputs=predictions)

# Compile model
model_final.compile(loss=keras.losses.categorical_crossentropy,
                    optimizer=keras.optimizers.Adadelta(),
                    metrics=['accuracy'])

# Data augmentation and normalization
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Generate training and validation data from directories
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',  # Ensure class_mode is set to 'categorical'
    shuffle=True  # Shuffle the training data
)

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',  # Ensure class_mode is set to 'categorical'
    shuffle=False  # Do not shuffle the validation data
)

# Configure ModelCheckpoint to save the best model during training
checkpoint = ModelCheckpoint("vgg19_1.keras", monitor='val_accuracy', verbose=1, save_best_only=True,
                             save_weights_only=False, mode='auto')

# Train the model
model_final.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
    callbacks=[checkpoint]
)

# Evaluate the model
score = model_final.evaluate(validation_generator)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
