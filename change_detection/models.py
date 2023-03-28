import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import *
from tensorflow.keras.utils import Sequence
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Convolution2D, ZeroPadding2D, MaxPooling2D, Conv2D, SeparableConv2D, Conv2DTranspose
from tensorflow.keras.layers import Input, Add, Dropout, Permute
from tensorflow.keras.activations import sigmoid, softmax, relu
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.losses import CategoricalCrossentropy, BinaryCrossentropy
from tensorflow.keras.metrics import Accuracy, Precision, Recall, AUC, MeanIoU, BinaryAccuracy, binary_crossentropy
from tensorflow.python.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau


class Multi_Conv_Block(Layer):

  def __init__(self, num_filters=[16,16], kernel_size=3):
    super().__init__()

    self.multi_conv= Sequential()

    for filters in num_filters:
      self.multi_conv.add(Conv2D(filters, kernel_size, padding='same'))
      self.multi_conv.add(BatchNormalization())
  
  def call(self, x):
    return self.multi_conv(x)

class DoubleConv(Layer):
    def __init__(self, filters, kernel_size):
      super(DoubleConv, self).__init__()

      self.conv_layers= Sequential()
      self.conv_layers.add(Conv2D(filters, kernel_size, padding='same'))
      self.conv_layers.add(BatchNormalization())
      self.conv_layers.add(ReLU())
      self.conv_layers.add(Conv2D(filters, kernel_size, padding='same'))
      self.conv_layers.add(BatchNormalization())
      self.conv_layers.add(ReLU())

    def call(self, x):
        return self.conv_layers(x)
        
class UNET(Model):

  def __init__(self, num_classes=1, features=[64, 128, 256, 512, 1024], kernel_size=3):
    super(UNET, self).__init__()
    self.ups = []
    self.downs = []
    self.pool= MaxPooling2D(pool_size=2, strides=2, padding='same')

    # Downsampling 

    for feature in features:
      self.downs.append(DoubleConv(feature, kernel_size))

    # Upsampling

    for feature in features[::-1]:
      self.ups.append(Conv2DTranspose(feature, kernel_size=2, strides=2))
      self.ups.append(DoubleConv(feature, kernel_size))

    self.bottleneck= DoubleConv(features[-1], kernel_size)
    
    if(num_classes==1):
        self.final_conv= Conv2D(num_classes, 1, padding='same', activation=sigmoid)
    else:
        self.final_conv= Conv2D(num_classes, 1, padding='same', activation=softmax)
    
  
  def get_summary(self, input_shape):
    inputs=Input(shape=input_shape)
    return Model(inputs=[inputs], outputs=[self.call(inputs)]).summary()

  def call(self, x):
    skip_connections= []

    for down in self.downs:
      x= down(x)
      skip_connections.append(x)
      x= self.pool(x)
    
    x= self.bottleneck(x)
    skip_connections= skip_connections[::-1]

    for i in range(0, len(self.ups), 2):
      x= self.ups[i](x)
      connection= skip_connections[i//2]
      concat= tf.concat([x, connection], axis=-1)
      x= self.ups[i+1](x)
    

    return self.final_conv(x)


class Siamese_UNET(Model):
  def __init__(self, in_channels=6, num_classes=1,
               down_features= [[16, 16], [32, 32], [64, 64, 64], [128, 128, 128]],
               up_features=[[128, 128, 64, 64], [64, 64, 32, 32], [32, 16, 16], [16]],
               kernel_size=3,
               mode='concat'):
    super().__init__()
    
    assert mode in ['concat', 'diff']
    self.mode= mode
    self.ups = []
    self.downs = []
    self.in_channels= in_channels
    self.pool= MaxPooling2D(pool_size=2, strides=2, padding='same')
    self.bottle_neck= Sequential([
                                  Conv2D(128, kernel_size, padding='same'),
                                  Conv2DTranspose(128, kernel_size, strides=2, padding='same')])

    # Downsampling 

    for features in down_features:
      self.downs.append(Multi_Conv_Block(features, kernel_size))
    
    for features in up_features[:-1]:
      self.ups.append(Sequential([Multi_Conv_Block(features, kernel_size),
                    Conv2DTranspose(features[-1], kernel_size, strides=2, padding='same')])) 
    self.ups.append(Multi_Conv_Block(up_features[-1], kernel_size))
    
    if(num_classes==1):
        self.final_conv= Conv2D(num_classes, 1, padding='same', activation=sigmoid)
    else:
        self.final_conv= Conv2D(num_classes, 1, padding='same', activation=softmax)
        
  def call(self, x):
    
    x1, x2= x[:,:,:,:self.in_channels//2], x[:,:,:,self.in_channels//2:]
    skip_1= []
    skip_2= []

    for i, down in enumerate(self.downs):
      x1= down(x1)
      skip_1.append(x1)
      x1= self.pool(x1)
      x2= down(x2)
      skip_2.append(x2)

      if i!=(len(self.downs)-1):
        x2= self.pool(x2)
    
    x= self.bottle_neck(x1)
    skip_1= skip_1[::-1]
    skip_2= skip_2[::-1]
    
    for i in range(len(self.ups)):
      if self.mode is 'concat':
        x= tf.concat([x, skip_1[i], skip_2[i]], axis=-1)
      
      elif self.mode is 'diff':
        x= tf.concat([x, skip_1[i]-skip_2[i]], axis=-1)
      x= self.ups[i](x)

    return self.final_conv(x)

  def get_summary(self, input_shape):
    inputs=Input(shape=input_shape)
    return Model(inputs=[inputs], outputs=[self.call(inputs)]).summary()

def binary_focal_loss(y_true, y_pred,alpha=.25, gamma=2.0):
  y_true = tf.cast(y_true, tf.float32)
  epsilon = K.epsilon()
  # Add the epsilon to prediction value
  # y_pred = y_pred + epsilon
  # Clip the prediciton value
  y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
  # Calculate p_t
  p_t = tf.where(K.equal(y_true, 1), y_pred, 1 - y_pred)
  # Calculate alpha_t
  alpha_factor = K.ones_like(y_true) * alpha
  alpha_t = tf.where(K.equal(y_true, 1), alpha_factor, 1 - alpha_factor)
  # Calculate cross entropy
  cross_entropy = -K.log(p_t)
  weight = alpha_t * K.pow((1 - p_t), gamma)
  # Calculate focal loss
  loss = weight * cross_entropy
  # Sum the losses in mini_batch
  loss = K.mean(K.sum(loss, axis=1))
  return loss

def Dice_BCE_Loss(targets, inputs, smooth=1e-6, weights=[0.1, 0.9]):    
       
    #flatten label and prediction tensors
    inputs = K.flatten(inputs)
    targets = K.flatten(targets)
    
    BCE =  binary_crossentropy(targets, inputs)
    intersection = tf.reduce_sum(targets*inputs)   
    dice_loss = 1 - (2*intersection + smooth) / (K.sum(targets) + K.sum(inputs) + smooth)
    Dice_BCE = weights[0]*BCE + weights[1]*dice_loss
    
    return Dice_BCE