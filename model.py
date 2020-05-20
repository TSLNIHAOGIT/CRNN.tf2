from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input

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


def vgg16():
    base_model = VGG16(weights='imagenet', include_top=False,pooling=None)
    base_model.trainable = False
    vgg_model = keras.Sequential()

    # 将vgg16模型的 卷积层 添加到新模型中（不包含全连接层)
    ##-5是48可以自己加pooLling层变成24，也可以用默认的到-4是12
    for item in base_model.layers[:-5]:
        vgg_model.add(item)
    vgg_model.add(layers.MaxPool2D(pool_size=(2, 2), strides=(2, 1), padding='same'))


    '''
    或者是    
    for item in base_model.layers[:-4]:
        vgg_model.add(item)
    '''


    # vgg_model = keras.Sequential([
    #     base_model,
    #     # keras.layers.GlobalAveragePooling2D()
    # ])
    return vgg_model

print('vgg16 summary',vgg16().summary())

def crnn(num_classes):
    img_input = keras.Input(shape=(32, None, 3))

    # x = vgg(img_input)#self def vgg output x_shape=(None, 1, None, 512)
    x=vgg16()(img_input)
    print('x',x.shape)#base_model output x_shape= (None, 1, None, 512)


    x = layers.Reshape((-1, 512))(x)

    x = layers.Bidirectional(layers.LSTM(units=256, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(units=256, return_sequences=True))(x)
    x = layers.Dense(units=num_classes)(x)
    #self inputs=Tensor("input_2:0", shape=(None, 32, None, 3), dtype=float32), outputs=Tensor("dense/Identity:0", shape=(None, None, 63), dtype=float32)

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


'''
自己加pooling层：24
decoded ['7PFRN3SU', '45BQYZB', '79VE', '8DK', '2', 'MB7A', 'RAK', '4CED', '1VFYO', 'U430WE', '920F', 'PMS6', 'S', '3UDM8V', 'H', '9SWGYB', 'YZAEN9E', 'X', 'SQCBR53', 'N8FLG12Y', '3S5R1T6X', '89TDTV', 'FPJJTA4H', 'QPIPII1', 'CWFZL']
Epoch 49 Batch 10 Loss 3.2187  
y_shape=(25, 10),y_pred_shape=(25, 24, 63)
logit_length_shape=(25,)


#默认pooling层:12
decoded ['RDQK5', 'U430WE', '0UN', '6X5', 'G65', 'F7Y365', '7', 'S', 'X', 'QVEP0AS', '2S7XX', 'MJUADSXB', 'J4341', '3I6MG0', 'X9FGH31S9', 'WVS', '9TGR2BRG', '61MX98R4J', '7ULKVDAS', '5V92', 'VW', '3V5', 'T5W0KR9X', 'PMS6', 'RAK']
Epoch 49 Batch 10 Loss 2.9665  
y_shape=(25, 10),y_pred_shape=(25, 12, 63)
logit_length_shape=(25,)

'''