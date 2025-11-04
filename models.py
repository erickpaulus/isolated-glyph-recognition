from keras.callbacks import ReduceLROnPlateau
from keras.layers import (Input, Conv2D, BatchNormalization, ZeroPadding2D,AveragePooling2D,GlobalMaxPooling2D,GlobalAveragePooling2D,
                          MaxPooling2D, Activation, Add, Dense, Dropout, Flatten,DepthwiseConv2D, Concatenate, ReLU, LeakyReLU,Reshape, Lambda)
from keras.models import Model,Sequential
# from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.initializers import glorot_uniform

from keras import optimizers
# import efficientnet.tfkeras as enet
# import efficientnet.keras as enet

# Swish defination
# from keras.backend import sigmoid
from tensorflow.keras.backend import sigmoid
from tensorflow.keras.applications import ResNet50, ResNet101, ResNet152V2, InceptionResNetV2, MobileNet, MobileNetV2, NASNetMobile
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications import ResNet101V2, ResNet50V2 
from tensorflow.keras.applications import efficientnet_v2
from tensorflow.keras.applications.resnet_v2 import ResNet50V2, ResNet101V2
import tensorflow as tf


from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense, MultiHeadAttention, LayerNormalization, Add, Dropout,Multiply, Add, Softmax, Lambda


# start - Define the VGG16 model
def multi_conv_pool( x, filters, n):
    '''
    Builds (Conv2D - BN - Relu) X n - MaxPooling2D
    The training is regularized by global weight decay (5e-4) in the original paper,
    but BN is used here instead of weight decay
    Args:
        x - input
        output_channel - 
        n - number of convolution layer
        
    Returns:
        multi conv + max pooling block
    '''
    y = x
    for _ in range(n):
        y = Conv2D(filters, (3, 3), padding = 'same')(y)
        y = BatchNormalization()(y)
        y = Activation('relu')(y)
    y = MaxPooling2D(strides = (2, 2))(y)
    return y


def build_vgg16(input_shape,nb_classes):
    input = Input(shape = input_shape) #for input 2D    
    y = ZeroPadding2D(padding = (2, 2))(input) # matching the image size of CIFAR-10

    y = multi_conv_pool(y, 64, 2) # 32x32
    y = multi_conv_pool(y, 128, 2) # 16x16
    y = multi_conv_pool(y, 256, 3) # 8x8
    y = multi_conv_pool(y, 512, 3) # 4x4
    y = multi_conv_pool(y, 512, 3) # 2x2
    y = Flatten()(y)
    y = Dense(units = 256, activation='relu')(y) # original paper suggests 4096 FC
    y = Dropout(0.2)(y)
    y = Dense(units = 256, activation='relu')(y)
    y = Dropout(0.2)(y)
    
    output = Dense(nb_classes, activation='softmax')(y)
    model = Model(input, output)
    return  model

# model = build_vgg16(input_shape,nb_classes)
# model.summary()
# End - Define the VGG16 model

# Start - Define the ResNet model
def build_resnet(input_shape, nb_classes):
    input = Input(shape=input_shape)
    y = ZeroPadding2D(padding = (2, 2))(input) # matching the image size of CIFAR-10
    y = Conv2D(64, 7, padding="same")(y)
    y = BatchNormalization()(y)
    y = Activation("relu")(y)
    y = MaxPooling2D()(y)
    y = residual_block(y, 64)
    y = residual_block(y, 128, stride=2)
    y = residual_block(y, 256, stride=2)
    y = residual_block(y, 512, stride=2)
    y = GlobalAveragePooling2D()(y)
    output = Dense(nb_classes, activation="softmax")(y)
    
    model = Model(input, output)
    return model

# Define the residual block
def residual_block(y, filters, stride=1):
    shortcut = y
    y = Conv2D(filters, 3, strides=stride, padding="same")(y)
    y = BatchNormalization()(y)
    y = Activation("relu")(y)
    y = Conv2D(filters, 3, padding="same")(y)
    y = BatchNormalization()(y)
    if stride > 1:
        shortcut = Conv2D(filters, 1, strides=stride, padding="same")(shortcut)
        shortcut = BatchNormalization()(shortcut)
    y = Add()([y, shortcut]) 
    y = Activation("relu")(y)
    return y


# model = build_resnet(input_shape, nb_classes)
# model.summary()

# End - Define the ResNet model

# Start - Define the MobileNet model
def _depthwise_sep_conv(x, filters, alpha, strides = (1, 1)):
    '''
    Creates a depthwise separable convolution block

    Args:
        x - input
        filters - the number of output filters
        alpha - width multiplier
        strides - the stride length of the convolution

    Returns:
        A depthwise separable convolution block
    '''
    
    y = DepthwiseConv2D((3, 3), padding = 'same', strides = strides)(x)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = Conv2D(int(filters * alpha), (1, 1), padding = 'same')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    return y
      
def build_mobilenet(input_shape, nb_classes, alpha):   
    input = Input(shape = input_shape)
    y = ZeroPadding2D(padding = (2, 2))(input) # matching the image size of CIFAR-10

    # some layers have different strides from the papers considering the size of mnist
    y = Conv2D(int(32 * alpha), (3, 3), padding = 'same')(y) # strides = (2, 2) in the paper
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = _depthwise_sep_conv(y, 64, alpha) # spatial size: 32 x 32
    y = _depthwise_sep_conv(y, 128, alpha, strides = (2, 2)) # spatial size: 32 x 32
    y = _depthwise_sep_conv(y, 128, alpha) # spatial size: 16 x 16
    y = _depthwise_sep_conv(y, 256, alpha, strides = (2, 2)) # spatial size: 8 x 8
    y = _depthwise_sep_conv(y, 256, alpha) # spatial size: 8 x 8
    y = _depthwise_sep_conv(y, 512, alpha, strides = (2, 2)) # spatial size: 4 x 4
    for _ in range(5):
        y = _depthwise_sep_conv(y, 512, alpha) # spatial size: 4 x 4
    y = _depthwise_sep_conv(y, 1024, alpha, strides = (2, 2)) # spatial size: 2 x 2
    y = _depthwise_sep_conv(y, 1024, alpha) # strides = (2, 2) in the paper
    y = GlobalAveragePooling2D()(y)    
    output = Dense(nb_classes, activation="softmax")(y)

    model = Model(input, output)
    return model

# # Build the MobileNet model
# ALPHA = 1 # 0 < alpha <= 1
# alpha = ALPHA 
# model = build_mobilenet(input_shape, nb_classes, alpha)
# model.summary()
# End - Define the MobileNet model

# Start - Define the MobileNet model
def CNN_model(input_shape, nb_classes):
    model = Sequential()                                 # Linear stacking of layers

    # Convolution Layer 1
    model.add(Conv2D(32, (3, 3), input_shape=input_shape)) # 32 different 3x3 kernels -- so 32 feature maps
    model.add(BatchNormalization(axis=-1))               # normalize each feature map before activation
    convLayer01 = Activation('relu')                     # activation
    model.add(convLayer01)

    # Convolution Layer 2
    model.add(Conv2D(32, (3, 3)))                        # 32 different 3x3 kernels -- so 32 feature maps
    model.add(BatchNormalization(axis=-1))               # normalize each feature map before activation
    model.add(Activation('relu'))                        # activation
    convLayer02 = MaxPooling2D(pool_size=(2,2))          # Pool the max values over a 2x2 kernel
    model.add(convLayer02)

    # Convolution Layer 3
    model.add(Conv2D(64,(3, 3)))                         # 64 different 3x3 kernels -- so 64 feature maps
    model.add(BatchNormalization(axis=-1))               # normalize each feature map before activation
    convLayer03 = Activation('relu')                     # activation
    model.add(convLayer03)

    # Convolution Layer 4
    model.add(Conv2D(64, (3, 3)))                        # 64 different 3x3 kernels -- so 64 feature maps
    model.add(BatchNormalization(axis=-1))               # normalize each feature map before activation
    model.add(Activation('relu'))                        # activation
    convLayer04 = MaxPooling2D(pool_size=(2,2))          # Pool the max values over a 2x2 kernel
    model.add(convLayer04)
    model.add(Flatten())                                 # Flatten final 4x4x64 output matrix into a 1024-length vector

    # Fully Connected Layer 5
    model.add(Dense(512))                                # 512 FCN nodes
    model.add(BatchNormalization())                      # normalization
    model.add(Activation('relu'))                        # activation

    # Fully Connected Layer 6                       
    model.add(Dropout(0.2))                              # 20% dropout of randomly selected nodes
    model.add(Dense(nb_classes))                                 # final 10 FCN nodes
    model.add(Activation('softmax'))                     # softmax activation
    return model



# Start - Define the ResNet50 model
def identity_block(X, f, filters, stage, block):
    
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    F1, F2, F3 = filters
    
    X_shortcut = X
        
    X = Conv2D(filters = F1, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 1, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)
        
    X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1,1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 1, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 1, name = bn_name_base + '2c')(X)

    # Add shortcut value to main path
    X = Add()([X_shortcut, X])
    X = Activation('relu')(X)
        
    return X

def convolutional_block(X, f, filters, stage, block, s = 2):        
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    F1, F2, F3 = filters
    X_shortcut = X
    X = Conv2D(F1, (1, 1), strides = (s,s), name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 1, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)
    X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1,1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 1, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)
    X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 1, name = bn_name_base + '2c')(X)
    X_shortcut = Conv2D(filters = F3, kernel_size = (1, 1), strides = (s,s), padding = 'valid', name = conv_name_base + '1', kernel_initializer = glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis = 1, name = bn_name_base + '1')(X_shortcut)
    X = Add()([X_shortcut, X])
    X = Activation('relu')(X)
   
    return X

def build_resnet50(input_shape, nb_classes):
    input = Input(input_shape)
    X = ZeroPadding2D(padding = (2, 2))(input) 
    # X = ZeroPadding2D((3, 3))(input)
    X = Conv2D(64, (7, 7), strides = (2, 2), name = 'conv1', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 1, name = 'bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)
    X = convolutional_block(X, f = 3, filters = [64, 64, 256], stage = 2, block='a', s = 1)
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')
    X = convolutional_block(X, f = 3, filters = [128, 128, 512], stage = 3, block='a', s = 2)
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='b')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='c')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='d')
    X = convolutional_block(X, f = 3, filters = [256, 256, 1024], stage = 4, block='a', s = 2)
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='b')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='c')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='d')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='e')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='f')
    X = convolutional_block(X, f = 3, filters = [512, 512, 2048], stage = 5, block='a', s = 2)
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='b')
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='c')
    X = AveragePooling2D(pool_size=(2, 2),name='avg_pool')(X)
    X = Flatten()(X)
    X = Dense(nb_classes, activation='softmax', name='fc' + str(nb_classes), kernel_initializer = glorot_uniform(seed=0))(X)
    model = Model(inputs = input, outputs = X, name='ResNet50')
    return model
# End - Define the ResNet50 model

# Start - Define the MobileNetv2 model
def expansion_block(x,t,filters,block_id):
    prefix = 'block_{}_'.format(block_id)
    total_filters = t*filters
    x = Conv2D(total_filters,1,padding='same',use_bias=False, name = prefix +'expand')(x)
    x = BatchNormalization(name=prefix +'expand_bn')(x)
    x = ReLU(6,name = prefix +'expand_relu')(x)
    return x

def depthwise_block(x,stride,block_id):
    prefix = 'block_{}_'.format(block_id)
    x = DepthwiseConv2D(3,strides=(stride,stride),padding ='same', use_bias = False, name = prefix + 'depthwise_conv')(x)
    x = BatchNormalization(name=prefix +'dw_bn')(x)
    x = ReLU(6,name=prefix +'dw_relu')(x)
    return x

def projection_block(x,out_channels,block_id):
    prefix = 'block_{}_'.format(block_id)
    x = Conv2D(filters = out_channels,kernel_size = 1,padding='same',use_bias=False,name= prefix + 'compress')(x)
    x = BatchNormalization(name=prefix +'compress_bn')(x)
    return x

def Bottleneck(x,t,filters, out_channels,stride,block_id):
    y = expansion_block(x,t,filters,block_id)
    y = depthwise_block(y,stride,block_id)
    y = projection_block(y, out_channels,block_id)
    if y.shape[-1]==x.shape[-1]:
        y = Add()([x,y])
    return y


def build_mobilenetv2(input_shape, nb_classes):
    input = Input(shape = input_shape)
    x = ZeroPadding2D(padding = (2, 2))(input)

    x = Conv2D(32,kernel_size=3,strides=(2,2),padding = 'same', use_bias=False)(x)
    x = BatchNormalization(name='conv1_bn')(x)
    x = ReLU(6, name = 'conv1_relu')(x)

    # 17 Bottlenecks

    x = depthwise_block(x,stride=1,block_id=1)
    x = projection_block(x, out_channels=16,block_id=1)

    x = Bottleneck(x, t = 6, filters = x.shape[-1], out_channels = 24, stride = 2,block_id = 2)
    x = Bottleneck(x, t = 6, filters = x.shape[-1], out_channels = 24, stride = 1,block_id = 3)

    x = Bottleneck(x, t = 6, filters = x.shape[-1], out_channels = 32, stride = 2,block_id = 4)
    x = Bottleneck(x, t = 6, filters = x.shape[-1], out_channels = 32, stride = 1,block_id = 5)
    x = Bottleneck(x, t = 6, filters = x.shape[-1], out_channels = 32, stride = 1,block_id = 6)

    x = Bottleneck(x, t = 6, filters = x.shape[-1], out_channels = 64, stride = 2,block_id = 7)
    x = Bottleneck(x, t = 6, filters = x.shape[-1], out_channels = 64, stride = 1,block_id = 8)
    x = Bottleneck(x, t = 6, filters = x.shape[-1], out_channels = 64, stride = 1,block_id = 9)
    x = Bottleneck(x, t = 6, filters = x.shape[-1], out_channels = 64, stride = 1,block_id = 10)

    x = Bottleneck(x, t = 6, filters = x.shape[-1], out_channels = 96, stride = 1,block_id = 11)
    x = Bottleneck(x, t = 6, filters = x.shape[-1], out_channels = 96, stride = 1,block_id = 12)
    x = Bottleneck(x, t = 6, filters = x.shape[-1], out_channels = 96, stride = 1,block_id = 13)

    x = Bottleneck(x, t = 6, filters = x.shape[-1], out_channels = 160, stride = 2,block_id = 14)
    x = Bottleneck(x, t = 6, filters = x.shape[-1], out_channels = 160, stride = 1,block_id = 15)
    x = Bottleneck(x, t = 6, filters = x.shape[-1], out_channels = 160, stride = 1,block_id = 16)

    x = Bottleneck(x, t = 6, filters = x.shape[-1], out_channels = 320, stride = 1,block_id = 17)


    #1*1 conv
    x = Conv2D(filters = 1280,kernel_size = 1,padding='same',use_bias=False, name = 'last_conv')(x)
    x = BatchNormalization(name='last_bn')(x)
    x = ReLU(6,name='last_relu')(x)

    #AvgPool 7*7
    x = GlobalAveragePooling2D(name='global_average_pool')(x)

    output = Dense(nb_classes,activation='softmax')(x)

    model = Model(input, output)

    return model

# Start - Define the MobileNetv2 model
# what is  the swish activation?
def swish_act(x, beta = 1):
    return (x * sigmoid(beta * x))

# def build_EfficientNetV2B0_keras(input_shape, nb_classes):
#     # loading B0 pre-trained on ImageNet without final aka fiature extractor
#     model = enet.EfficientNetV2B0(include_top=False, input_shape=input_shape, pooling='avg', weights='imagenet')

#     # building 2 fully connected layer 
#     y = model.output

#     y = BatchNormalization()(y)
#     y = Dropout(0.2)(y)

#     y = Dense(512)(y)
#     y = BatchNormalization()(y)
#     y = Activation(swish_act)(y)
#     y = Dropout(0.2)(y)

#     y = Dense(256)(y)
#     y = BatchNormalization()(y)
#     y = Activation(swish_act)(y)

#     # output layer
#     output = Dense(nb_classes, activation="softmax")(y)

#     model_final = Model(inputs = model.input, outputs = output)
    
#     return model_final

# def build_EfficientNetB7_keras(input_shape, nb_classes):
#     # loading B0 pre-trained on ImageNet without final aka fiature extractor
#     model = enet.EfficientNetB0(include_top=False, input_shape=input_shape, pooling='avg', weights='imagenet')

#     # building 2 fully connected layer 
#     y = model.output

#     y = BatchNormalization()(y)
#     y = Dropout(0.2)(y)

#     y = Dense(512)(y)
#     y = BatchNormalization()(y)
#     y = Activation(swish_act)(y)
#     y = Dropout(0.2)(y)

#     y = Dense(256)(y)
#     y = BatchNormalization()(y)
#     y = Activation(swish_act)(y)

#     # output layer
#     output = Dense(nb_classes, activation="softmax")(y)

#     model_final = Model(inputs = model.input, outputs = output)
    
#     return model_final



def build_VGG19_keras(input_shape, nb_classes, weight = None):
    # loading B0 pre-trained on ImageNet without final aka fiature extractor
    model = VGG19(include_top=False, input_shape=input_shape, pooling='avg', weights= weight)
    model.trainable = True
    # building 2 fully connected layer 
    y = model.output

    y = BatchNormalization()(y)
    y = Dropout(0.2)(y)

    y = Dense(512)(y)
    y = BatchNormalization()(y)
    y = Activation(swish_act)(y)
    y = Dropout(0.2)(y)

    y = Dense(256)(y)
    y = BatchNormalization()(y)
    y = Activation(swish_act)(y)

    # output layer
    output = Dense(nb_classes, activation="softmax")(y)

    model_final = Model(inputs = model.input, outputs = output)
    
    return model_final


def build_ResNet50_keras(input_shape, nb_classes , weight = None):
    # loading B0 pre-trained on ImageNet without final aka fiature extractor
    model = ResNet50(include_top=False, input_shape=input_shape, pooling='avg', weights=weight)
    model.trainable = True
    # building 2 fully connected layer 
    y = model.output

    y = BatchNormalization()(y)
    y = Dropout(0.2)(y)

    y = Dense(512)(y)
    y = BatchNormalization()(y)
    y = Activation(swish_act)(y)
    y = Dropout(0.2)(y)

    y = Dense(256)(y)
    y = BatchNormalization()(y)
    y = Activation(swish_act)(y)

    # output layer
    output = Dense(nb_classes, activation="softmax")(y)

    model_final = Model(inputs = model.input, outputs = output)
    
    return model_final

def build_ResNet101_keras(input_shape, nb_classes, weight = None):
    # loading B0 pre-trained on ImageNet without final aka fiature extractor
    model = ResNet101(include_top=False, input_shape=input_shape, pooling='avg', weights=weight)
    model.trainable = True
    # building 2 fully connected layer 
    y = model.output

    y = BatchNormalization()(y)
    y = Dropout(0.2)(y)

    y = Dense(512)(y)
    y = BatchNormalization()(y)
    y = Activation(swish_act)(y)
    y = Dropout(0.2)(y)

    y = Dense(256)(y)
    y = BatchNormalization()(y)
    y = Activation(swish_act)(y)

    # output layer
    output = Dense(nb_classes, activation="softmax")(y)

    model_final = Model(inputs = model.input, outputs = output)
    
    return model_final

def build_ResNet152V2_keras(input_shape, nb_classes, weight = None):
    # loading B0 pre-trained on ImageNet without final aka fiature extractor
    model = ResNet152V2(include_top=False, input_shape=input_shape, pooling='avg', weights=weight)
    model.trainable = True
    # building 2 fully connected layer 
    y = model.output

    y = BatchNormalization()(y)
    y = Dropout(0.2)(y)

    y = Dense(512)(y)
    y = BatchNormalization()(y)
    y = Activation(swish_act)(y)
    y = Dropout(0.2)(y)

    y = Dense(256)(y)
    y = BatchNormalization()(y)
    y = Activation(swish_act)(y)

    # output layer
    output = Dense(nb_classes, activation="softmax")(y)

    model_final = Model(inputs = model.input, outputs = output)
    
    return model_final

def build_InceptionResNetV2_keras(input_shape, nb_classes, weight = None):
    # loading B0 pre-trained on ImageNet without final aka fiature extractor
    model = InceptionResNetV2(include_top=False, input_shape=input_shape, pooling='avg', weights=weight)
    model.trainable = True
    # building 2 fully connected layer 
    y = model.output

    y = BatchNormalization()(y)
    y = Dropout(0.2)(y)

    y = Dense(512)(y)
    y = BatchNormalization()(y)
    y = Activation(swish_act)(y)
    y = Dropout(0.2)(y)

    y = Dense(256)(y)
    y = BatchNormalization()(y)
    y = Activation(swish_act)(y)

    # output layer
    output = Dense(nb_classes, activation="softmax")(y)

    model_final = Model(inputs = model.input, outputs = output)
    
    return model_final

# def build_MobileNet_keras(input_shape, nb_classes):
#     base_model = MobileNet(include_top=False, input_shape=input_shape, weights='imagenet')
#     x = base_model.output
#     x = GlobalAveragePooling2D()(x)
#     # Add your custom layers here
#     x = Dense(512, activation='relu')(x)
#     x = Dense(128, activation='relu')(x)
#     predictions = Dense(nb_classes, activation='softmax')(x)
#     model_final = Model(inputs=base_model.input, outputs=predictions)
#     # Load the saved weights into the transfer learning model
#     model_final.load_weights('./results/TB-MobileNet-size(75, 75, 3)-batch32-improvement-epoch31-val_accuracy0.9550_weight.hdf5', by_name=True)

#     # Freeze the base model layers to prevent them from being trained
#     for layer in base_model.layers:
#         layer.trainable = False
#     return model_final

def build_TransferLearning_MobileNet_keras(input_shape, nb_classes):
    # loading B0 pre-trained on ImageNet without final aka fiature extractor
    model = MobileNet(include_top=False, input_shape=input_shape, pooling='avg', weights=None)
    # Load the saved weights into the transfer learning model
    model.load_weights('./results/TB_gray-MobileNet-size(75, 75, 3)-batch32-improvement-epoch20-val_accuracy0.9556_weight.hdf5', by_name=True)

    # Freeze the base model layers to prevent them from being trained
    # for layer in model.layers:
    #      layer.trainable = False
    
    # building 2 fully connected layer 
    y = model.output
    
    y = BatchNormalization()(y)
    y = Dropout(0.2)(y)

    y = Dense(512)(y)
    y = BatchNormalization()(y)
    y = Activation(swish_act)(y)
    y = Dropout(0.2)(y)

    y = Dense(256)(y)
    y = BatchNormalization()(y)
    y = Activation(swish_act)(y)

    # output layer
    output = Dense(nb_classes, activation="softmax")(y)

    model_final = Model(inputs = model.input, outputs = output)
    
    return model_final

def build_MobileNet_keras(input_shape, nb_classes, weight=None):
    # loading B0 pre-trained on ImageNet without final aka fiature extractor
    model = MobileNet(include_top=False, input_shape=input_shape, pooling='avg', weights=weight)
    
    model.trainable = True
    # building 2 fully connected layer 
    y = model.output
    
    y = BatchNormalization()(y)
    y = Dropout(0.2)(y)

    y = Dense(512)(y)
    y = BatchNormalization()(y)
    y = Activation(swish_act)(y)
    y = Dropout(0.2)(y)

    y = Dense(256)(y)
    y = BatchNormalization()(y)
    y = Activation(swish_act)(y)

    # output layer
    output = Dense(nb_classes, activation="softmax")(y)

    model_final = Model(inputs = model.input, outputs = output)
    
    return model_final

def build_inference_MobileNet_keras(input_shape, nb_classes,weight_path):
    # loading B0 pre-trained on ImageNet without final aka fiature extractor
    model = MobileNet( input_shape=input_shape, pooling='avg', weights=weight_path)
    
    model.trainable = True
    # building 2 fully connected layer 
    y = model.output
    
    # y = BatchNormalization()(y)
    # y = Dropout(0.2)(y)

    # y = Dense(512)(y)
    # y = BatchNormalization()(y)
    # y = Activation(swish_act)(y)
    # y = Dropout(0.2)(y)

    # y = Dense(256)(y)
    # y = BatchNormalization()(y)
    # y = Activation(swish_act)(y)

    # # output layer
    # output = Dense(nb_classes, activation="softmax")(y)

    # model_final = Model(inputs = model.input, outputs = output)
    
    return model

def build_MobileNetV2_keras(input_shape, nb_classes, weight = None):
    # loading B0 pre-trained on ImageNet without final aka fiature extractor
    model = MobileNetV2(include_top=False, input_shape=input_shape, pooling='avg', weights=weight)
    model.trainable = True
    # building 2 fully connected layer 
    y = model.output

    y = BatchNormalization()(y)
    y = Dropout(0.2)(y)

    y = Dense(512)(y)
    y = BatchNormalization()(y)
    y = Activation(swish_act)(y)
    y = Dropout(0.2)(y)

    y = Dense(256)(y)
    y = BatchNormalization()(y)
    y = Activation(swish_act)(y)

    # output layer
    output = Dense(nb_classes, activation="softmax")(y)

    model_final = Model(inputs = model.input, outputs = output)
    
    return model_final

def build_NASNetMobile_keras(input_shape, nb_classes, weight = None):
    # loading B0 pre-trained on ImageNet without final aka fiature extractor
    model = NASNetMobile(include_top=False, input_shape=input_shape, pooling='avg', weights=weight)
    model.trainable = True
    # building 2 fully connected layer 
    y = model.output

    y = BatchNormalization()(y)
    y = Dropout(0.2)(y)

    y = Dense(512)(y)
    y = BatchNormalization()(y)
    y = Activation(swish_act)(y)
    y = Dropout(0.2)(y)

    y = Dense(256)(y)
    y = BatchNormalization()(y)
    y = Activation(swish_act)(y)

    # output layer
    output = Dense(nb_classes, activation="softmax")(y)

    model_final = Model(inputs = model.input, outputs = output)
    
    return model_final


def build_ResNet50V2_keras(input_shape, nb_classes, weight = None):
    # loading B0 pre-trained on ImageNet without final aka fiature extractor
    model = ResNet50V2(include_top=False, input_shape=input_shape, pooling='avg', weights=weight)
    model.trainable = True
    y = model.output

    y = BatchNormalization()(y)
    y = Dropout(0.2)(y)

    y = Dense(512)(y)
    y = BatchNormalization()(y)
    y = Activation(swish_act)(y)
    y = Dropout(0.2)(y)

    y = Dense(256)(y)
    y = BatchNormalization()(y)
    y = Activation(swish_act)(y)

    # output layer
    output = Dense(nb_classes, activation="softmax")(y)

    model_final = Model(inputs = model.input, outputs = output)
    
    return model_final


def build_ResNet101V2_keras(input_shape, nb_classes, weight = None):
    # loading B0 pre-trained on ImageNet without final aka fiature extractor
    model = ResNet101V2(include_top=False, input_shape=input_shape, pooling='avg', weights=weight)
    model.trainable = True
    # building 2 fully connected layer 
    y = model.output

    y = BatchNormalization()(y)
    y = Dropout(0.2)(y)

    y = Dense(512)(y)
    y = BatchNormalization()(y)
    y = Activation(swish_act)(y)
    y = Dropout(0.2)(y)

    y = Dense(256)(y)
    y = BatchNormalization()(y)
    y = Activation(swish_act)(y)

    # output layer
    output = Dense(nb_classes, activation="softmax")(y)

    model_final = Model(inputs = model.input, outputs = output)
    
    return model_final

def build_EfficientNetV2B0_keras(input_shape, nb_classes, weight = None):
    # loading B0 pre-trained on ImageNet without final aka fiature extractor
    model = efficientnet_v2.EfficientNetV2B0(include_top=False, input_shape=input_shape, pooling='avg', weights=weight)
    model.trainable = True
    # building 2 fully connected layer 
    y = model.output

    y = BatchNormalization()(y)
    y = Dropout(0.2)(y)

    y = Dense(512)(y)
    y = BatchNormalization()(y)
    y = Activation(swish_act)(y)
    y = Dropout(0.2)(y)

    y = Dense(256)(y)
    y = BatchNormalization()(y)
    y = Activation(swish_act)(y)

    # output layer
    output = Dense(nb_classes, activation="softmax")(y)

    model_final = Model(inputs = model.input, outputs = output)
    
    return model_final

def build_EfficientNetV2B3_keras(input_shape, nb_classes, weight = None):
    # loading B0 pre-trained on ImageNet without final aka fiature extractor
    model = efficientnet_v2.EfficientNetV2B3(include_top=False, input_shape=input_shape, pooling='avg', weights=weight)
    model.trainable = True
    # building 2 fully connected layer 
    y = model.output

    y = BatchNormalization()(y)
    y = Dropout(0.2)(y)

    y = Dense(512)(y)
    y = BatchNormalization()(y)
    y = Activation(swish_act)(y)
    y = Dropout(0.2)(y)

    y = Dense(256)(y)
    y = BatchNormalization()(y)
    y = Activation(swish_act)(y)

    # output layer
    output = Dense(nb_classes, activation="softmax")(y)

    model_final = Model(inputs = model.input, outputs = output)
    
    return model_final

def build_EfficientNetV2S_keras(input_shape, nb_classes, weight = None):
    # loading B0 pre-trained on ImageNet without final aka fiature extractor
    model = efficientnet_v2.EfficientNetV2S(include_top=False, input_shape=input_shape, pooling='avg', weights=weight)
    model.trainable = True
    # building 2 fully connected layer 
    y = model.output

    y = BatchNormalization()(y)
    y = Dropout(0.2)(y)

    y = Dense(512)(y)
    y = BatchNormalization()(y)
    y = Activation(swish_act)(y)
    y = Dropout(0.2)(y)

    y = Dense(256)(y)
    y = BatchNormalization()(y)
    y = Activation(swish_act)(y)

    # output layer
    output = Dense(nb_classes, activation="softmax")(y)

    model_final = Model(inputs = model.input, outputs = output)
    
    return model_final

def build_EfficientNetV2M_keras(input_shape, nb_classes, weight = None):
    # loading B0 pre-trained on ImageNet without final aka fiature extractor
    model = efficientnet_v2.EfficientNetV2M(include_top=False, input_shape=input_shape, pooling='avg', weights=weight)
    model.trainable = True
    # building 2 fully connected layer 
    y = model.output

    y = BatchNormalization()(y)
    y = Dropout(0.2)(y)

    y = Dense(512)(y)
    y = BatchNormalization()(y)
    y = Activation(swish_act)(y)
    y = Dropout(0.2)(y)

    y = Dense(256)(y)
    y = BatchNormalization()(y)
    y = Activation(swish_act)(y)

    # output layer
    output = Dense(nb_classes, activation="softmax")(y)

    model_final = Model(inputs = model.input, outputs = output)
    
    return model_final

def build_EfficientNetV2L_keras(input_shape, nb_classes, weight = None):
    # loading B0 pre-trained on ImageNet without final aka fiature extractor
    model = efficientnet_v2.EfficientNetV2L(include_top=False, input_shape=input_shape, pooling='avg', weights=weight)
    model.trainable = True
    # building 2 fully connected layer 
    y = model.output

    y = BatchNormalization()(y)
    y = Dropout(0.2)(y)

    y = Dense(512)(y)
    y = BatchNormalization()(y)
    y = Activation(swish_act)(y)
    y = Dropout(0.2)(y)

    y = Dense(256)(y)
    y = BatchNormalization()(y)
    y = Activation(swish_act)(y)

    # output layer
    output = Dense(nb_classes, activation="softmax")(y)

    model_final = Model(inputs = model.input, outputs = output)
    
    return model_final


def build_MobiEff_keras(input_shape, nb_classes, weight='imagenet'):
    # Shared input for both models
    shared_input = Input(shape=input_shape)

    # Load pre-trained models (without top layers)
    model1 = MobileNet(include_top=False, input_shape=input_shape, pooling='avg', weights=weight)
    model2 = efficientnet_v2.EfficientNetV2S(include_top=False, input_shape=input_shape, pooling='avg', weights=weight)

    # Get features from shared input
    y1 = model1(shared_input)  # Output shape: (None, 1024)
    y2 = model2(shared_input)  # Output shape: (None, 1280)

    # Reduce dimensionality for consistency
    y1 = Dense(512, activation="swish")(y1)  
    y2 = Dense(512, activation="swish")(y2)

    # Concatenate feature outputs
    y = Concatenate()([y1, y2])  # Shape: (None, 1024)

    # Fully connected layers
    y = BatchNormalization()(y)
    y = Dropout(0.2)(y)

    y = Dense(256, activation="swish")(y)
    y = BatchNormalization()(y)

    # Output layer
    output = Dense(nb_classes, activation="softmax")(y)

    # Define final model with a single input
    model_final = Model(inputs=shared_input, outputs=output)

    return model_final

def build_IncResnetV2EfV2S_keras(input_shape, nb_classes, weight='imagenet'):
    # Shared input for both models
    shared_input = Input(shape=input_shape)

    # Load pre-trained models (without top layers)
    model1 = InceptionResNetV2(include_top=False, input_shape=input_shape, pooling='avg', weights=weight)
    model2 = efficientnet_v2.EfficientNetV2S(include_top=False, input_shape=input_shape, pooling='avg', weights=weight)

    # Get features from shared input
    y1 = model1(shared_input)  # Output shape: (None, 1024)
    y2 = model2(shared_input)  # Output shape: (None, 1280)

    # Reduce dimensionality for consistency
    y1 = Dense(512, activation="swish")(y1)  
    y2 = Dense(512, activation="swish")(y2)

    # Concatenate feature outputs
    y = Concatenate()([y1, y2])  # Shape: (None, 1024)

    # Fully connected layers
    y = BatchNormalization()(y)
    y = Dropout(0.2)(y)

    y = Dense(256, activation="swish")(y)
    y = BatchNormalization()(y)

    # Output layer
    output = Dense(nb_classes, activation="softmax")(y)

    # Define final model with a single input
    model_final = Model(inputs=shared_input, outputs=output)

    return model_final

# Define Cross-Attention Block
def cross_attention_block(query, key_value, num_heads=4, name="cross_attention"):
    """Applies Multi-Head Cross-Attention between feature maps."""
    attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=64, name=name)(query, key_value)
    attn_output = Add()([query, attn_output])  # Residual Connection
    attn_output = LayerNormalization()(attn_output)  # Normalize
    return attn_output

def build_CrossAttentionFusion_keras(input_shape, nb_classes, weight='imagenet'):
    # Define a single input
    
    input_tensor = Input(shape=input_shape)

    # Load Pretrained Models (Without Classification Layers)
    mobilenet = MobileNetV2(weights=weight, include_top=False, input_tensor=input_tensor)
    efficientnet = efficientnet_v2.EfficientNetV2S(weights=weight, include_top=False, input_tensor=input_tensor)

    # Extract Feature Maps
    mobile_features = GlobalAveragePooling2D()(mobilenet.output)  # Shape (Batch, Features)
    efficient_features = GlobalAveragePooling2D()(efficientnet.output)  # Shape (Batch, Features)

    # Reshape to Sequence Format for Transformer (Batch, 1, Features)
    # mobile_features = tf.expand_dims(mobile_features, axis=1)
    # efficient_features = tf.expand_dims(efficient_features, axis=1)
    # ✅ FIX: Use Lambda() to Expand Dimensions for Transformer Input
    mobile_features = Lambda(lambda x: tf.expand_dims(x, axis=1))(mobile_features)
    efficient_features = Lambda(lambda x: tf.expand_dims(x, axis=1))(efficient_features)


    # Cross-Attention: MobileNet attends to EfficientNet & vice versa
    attn_mobile = cross_attention_block(mobile_features, efficient_features, name="mobile_to_efficient")
    attn_efficient = cross_attention_block(efficient_features, mobile_features, name="efficient_to_mobile")

    # Fusion: Add both attention outputs
    fused_features = Add()([attn_mobile, attn_efficient])
    fused_features = LayerNormalization()(fused_features)

    # ✅ FIX: Use Lambda() to Squeeze the Sequence Dimension
    fused_features = Lambda(lambda x: tf.squeeze(x, axis=1))(fused_features)

    # Classification Head
    x = Dense(512, activation='relu')(fused_features)
    x = Dropout(0.5)(x)
    output = Dense(nb_classes, activation='softmax')(x)  # Change 10 to number of classes

    # Create Model
    model_final = Model(inputs=input_tensor, outputs=output)
    

    return model_final
    # Model Summary

# def build_AdaptiveCrossAttentionFusion_keras(input_shape, nb_classes, weight='imagenet'):
#     # Define a single input
#     input_tensor = Input(shape=input_shape)

#     # Load Pretrained Models (Without Classification Head)
#     mobilenet = MobileNetV2(weights=weight, include_top=False, input_tensor=input_tensor)
#     mobilenet.trainable = True
#     efficientnet = efficientnet_v2.EfficientNetV2S(weights=weight, include_top=False, input_tensor=input_tensor)
#     efficientnet.trainable = True
    
#     # Extract Feature Maps
#     mobile_features = GlobalAveragePooling2D()(mobilenet.output)  # Shape: (Batch, Features)
#     efficient_features = GlobalAveragePooling2D()(efficientnet.output)  # Shape: (Batch, Features)

#     # ✅ FIX: Use Keras Concatenate instead of tf.concat
#     concat_features = Concatenate()([mobile_features, efficient_features])

#     # Learnable Weights for Fusion
#     fusion_weights = Dense(2, activation="softmax")(concat_features)  # Softmax weights
#     alpha = Lambda(lambda x: x[:, 0:1])(fusion_weights)  # First weight
#     beta = Lambda(lambda x: x[:, 1:2])(fusion_weights)  # Second weight

#     # Adaptive Fusion
#     weighted_mobile = Multiply()([alpha, mobile_features])
#     weighted_efficient = Multiply()([beta, efficient_features])
#     fused_features = Add()([weighted_mobile, weighted_efficient])  # Final fused representation

#     # Classification Head
#     x = Dense(512, activation='relu')(fused_features)
#     x = tf.keras.layers.Dropout(0.5)(x)
#     output = Dense(nb_classes, activation='softmax')(x)  # Change 10 to the number of classes

#     # Create Model
#     model_final = Model(inputs=input_tensor, outputs=output)
    

#     return model_final

from tensorflow.keras import Input, Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Concatenate, Dense, Lambda, Multiply, Add
from tensorflow.keras.applications import EfficientNetV2S, EfficientNetV2M, EfficientNetV2L
import tensorflow as tf

def build_AdaptiveCrossAttentionFusion_keras(input_shape, nb_classes, weight='imagenet'):
    input_tensor = Input(shape=input_shape)

    # Load Pretrained Models (Without Classification Head)
    efficientnet_s = EfficientNetV2S(weights=weight, include_top=False, input_shape=input_shape)
    efficientnet_s.trainable = True
    efficientnet_m = EfficientNetV2M(weights=weight, include_top=False, input_shape=input_shape)
    efficientnet_m.trainable = True
    efficientnet_l = EfficientNetV2L(weights=weight, include_top=False, input_shape=input_shape)
    efficientnet_l.trainable = True

    # Extract Feature Maps
    efficientnet_s_features = GlobalAveragePooling2D()(efficientnet_s(input_tensor))
    efficientnet_m_features = GlobalAveragePooling2D()(efficientnet_m(input_tensor))
    efficientnet_l_features = GlobalAveragePooling2D()(efficientnet_l(input_tensor))

    # Concatenate the features from all three EfficientNetV2 versions
    concat_features = Concatenate()([efficientnet_s_features, efficientnet_m_features, efficientnet_l_features])

    # Learnable Weights for Fusion
    fusion_weights = Dense(3, activation="softmax")(concat_features)
    alpha_s = Lambda(lambda x: x[:, 0:1])(fusion_weights)
    alpha_m = Lambda(lambda x: x[:, 1:2])(fusion_weights)
    alpha_l = Lambda(lambda x: x[:, 2:3])(fusion_weights)

    # Adaptive Fusion
    weighted_s = Multiply()([alpha_s, efficientnet_s_features])
    weighted_m = Multiply()([alpha_m, efficientnet_m_features])
    weighted_l = Multiply()([alpha_l, efficientnet_l_features])
    fused_features = Add()([weighted_s, weighted_m, weighted_l])

    # Classification Head
    x = Dense(512, activation='relu')(fused_features)
    x = tf.keras.layers.Dropout(0.5)(x)
    output = Dense(nb_classes, activation='softmax')(x)

    # Create Model
    model_final = Model(inputs=input_tensor, outputs=output)

    return model_final


def build_AdaptiveCrossAttentionFusion_MobilenetV2_InceptionResNetV2_keras(input_shape, nb_classes, weight='imagenet'):
    # Define a single input
    input_tensor = Input(shape=input_shape)

    # Load Pretrained Models (Without Classification Head)
    mobilenet = MobileNetV2(weights=weight, include_top=False, input_tensor=input_tensor)
    mobilenet.trainable = True
    resnet = InceptionResNetV2(weights=weight, include_top=False, input_tensor=input_tensor)
    resnet.trainable = True

    # Extract Feature Maps
    # Extract and project features
    mobile_features = GlobalAveragePooling2D()(mobilenet.output)   # (None, 1280)
    resnet_features = GlobalAveragePooling2D()(resnet.output)      # (None, 2048)

    # Project to same dimension
    mobile_projected = Dense(512)(mobile_features)
    resnet_projected = Dense(512)(resnet_features)

    # Learnable fusion weights
    concat_features = Concatenate()([mobile_projected, resnet_projected])
    fusion_weights = Dense(2, activation="softmax")(concat_features)
    alpha = Lambda(lambda x: x[:, 0:1])(fusion_weights)
    beta = Lambda(lambda x: x[:, 1:2])(fusion_weights)

    # Adaptive fusion
    weighted_mobile = Multiply()([alpha, mobile_projected])
    weighted_resnet = Multiply()([beta, resnet_projected])
    fused_features = Add()([weighted_mobile, weighted_resnet])


    # Classification Head
    x = Dense(512, activation='relu')(fused_features)
    x = Dropout(0.5)(x)
    output = Dense(nb_classes, activation='softmax')(x)

    # Create Model
    model_final = Model(inputs=input_tensor, outputs=output)

    return model_final