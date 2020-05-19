from tensorflow import keras
from tensorflow.keras import layers


def vgg(input_tensor):
    """
    The original feature extraction structure from CRNN paper.
    Related paper: http://arxiv.org/abs/1507.05717
    """
    x = layers.Conv2D(
        filters=64, 
        kernel_size=3, 
        padding='same',
        activation='relu')(input_tensor)
    x = layers.MaxPool2D(pool_size=2, padding='same')(x)

    x = layers.Conv2D(
        filters=128, 
        kernel_size=3, 
        padding='same',
        activation='relu')(x)
    x = layers.MaxPool2D(pool_size=2, padding='same')(x)

    for i in range(2):
        x = layers.Conv2D(filters=256, kernel_size=3, padding='same',
                          activation='relu')(x)
    x = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 1), padding='same')(x)

    for i in range(2):
        x = layers.Conv2D(filters=512, kernel_size=3, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
    x = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 1), padding='same')(x)

    x = layers.Conv2D(filters=512, kernel_size=2, activation='relu')(x)
    return x


def crnn(num_classes):
    img_input = keras.Input(shape=(32, None, 1))

    x = vgg(img_input)
    x = layers.Reshape((-1, 512))(x)

    x = layers.Bidirectional(layers.LSTM(units=256, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(units=256, return_sequences=True))(x)
    x = layers.Dense(units=num_classes)(x)
    print('inputs={}, outputs={}'.format(img_input,x))
    return keras.Model(inputs=img_input, outputs=x, name='CRNN')


if __name__ == "__main__":
    model = crnn(10)
    model.summary()

    '''
    Model: "CRNN"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         [(None, 32, None, 1)]     0         
_________________________________________________________________
conv2d (Conv2D)              (None, 32, None, 64)      640       
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 16, None, 64)      0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 16, None, 128)     73856     
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 8, None, 128)      0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 8, None, 256)      295168    
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 8, None, 256)      590080    
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 4, None, 256)      0         
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 4, None, 512)      1180160   
_________________________________________________________________
batch_normalization (BatchNo (None, 4, None, 512)      2048      
_________________________________________________________________
activation (Activation)      (None, 4, None, 512)      0         
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 4, None, 512)      2359808   
_________________________________________________________________
batch_normalization_1 (Batch (None, 4, None, 512)      2048      
_________________________________________________________________
activation_1 (Activation)    (None, 4, None, 512)      0         
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 2, None, 512)      0         
_________________________________________________________________
conv2d_6 (Conv2D)            (None, 1, None, 512)      1049088   
_________________________________________________________________
reshape (Reshape)            (None, None, 512)         0         
_________________________________________________________________
bidirectional (Bidirectional (None, None, 512)         1574912   
_________________________________________________________________
bidirectional_1 (Bidirection (None, None, 512)         1574912   
_________________________________________________________________
dense (Dense)                (None, None, 10)          5130      
=================================================================
Total params: 8,707,850
Trainable params: 8,705,802
Non-trainable params: 2,048

    
    '''