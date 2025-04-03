### Importing libraries
from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, Flatten, Dense, 
                                     Dropout, GlobalAveragePooling2D, Concatenate)
from tensorflow.keras.models import Model
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Preprocessing
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_CLASSES = 10  # Change this based on your dataset

# Rescaling
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

#### Load the train dataset
train_generator = train_datagen.flow_from_directory(
    r'C:\Users\adity\Desktop\tomato disease dataset\tomato\val',  # Change to your dataset directory
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

#### split dataset into test
val_generator = train_datagen.flow_from_directory(
    r'C:\Users\adity\Desktop\tomato disease dataset\tomato\val',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# Input Layer
input_layer = Input(shape=(IMG_SIZE, IMG_SIZE, 3))

# Custom CNN Branch
x = Conv2D(32, (3,3), activation='relu', padding='same')(input_layer)
x = MaxPooling2D((2,2))(x)
x = Conv2D(64, (3,3), activation='relu', padding='same')(x)
x = MaxPooling2D((2,2))(x)
x = Flatten()(x)

# EfficientNet Branch
efficient_net = EfficientNetB0(include_top=False, weights='imagenet', input_tensor=input_layer)
efficient_net.trainable = False  # Freeze EfficientNet layers
efficient_output = GlobalAveragePooling2D()(efficient_net.output)

# Merge Features from CNN and EfficientNet
merged = Concatenate()([x, efficient_output])
merged = Dense(128, activation='relu')(merged)
merged = Dropout(0.5)(merged)
output_layer = Dense(NUM_CLASSES, activation='softmax')(merged)

# Build Model
hybrid_model = Model(inputs=input_layer, outputs=output_layer)

# Compile Model
hybrid_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Model Summary
hybrid_model.summary()

# Train Model
EPOCHS = 3

history=hybrid_model.fit(train_generator, validation_data=val_generator, epochs=EPOCHS)

# Save Model
hybrid_model.save("hybrid_cnn_eff_model.h5")

# Plot Accuracy
plt.figure(figsize=(8, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy', marker='o')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='o')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training & Validation Accuracy CNN_ Efficientnet')
plt.legend()
plt.grid(True)
plt.show()

# Plot Loss
plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'], label='Training Loss', marker='o', color='red')
plt.plot(history.history['val_loss'], label='Validation Loss', marker='o', color='blue')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training & Validation Loss CNN_ Efficientnet')
plt.legend()
plt.grid(True)
plt.show()

