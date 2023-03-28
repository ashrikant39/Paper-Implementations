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
from tensorflow.python.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

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

class Nested_UNet(Model):
  def __init__(self, num_classes=1,nb_filter=[32, 64, 128, 256, 512], kernel_size=3):
    super(Nested_UNet, self).__init__()
    self.nf = nb_filter
    self.pool= MaxPooling2D(pool_size=2, strides=2, padding='same')

    # All Downsampling convolution block
    self.conv0_0 = DoubleConv(nb_filter[0], kernel_size)
    self.conv1_0 = DoubleConv(nb_filter[1], kernel_size)
    self.conv2_0 = DoubleConv(nb_filter[2], kernel_size)
    self.conv3_0 = DoubleConv(nb_filter[3], kernel_size)
    self.conv4_0 = DoubleConv(nb_filter[4], kernel_size)

    self.conv0_1 = DoubleConv(nb_filter[0], kernel_size)
    self.conv1_1 = DoubleConv(nb_filter[1], kernel_size)
    self.conv2_1 = DoubleConv(nb_filter[2], kernel_size)
    self.conv3_1 = DoubleConv(nb_filter[3], kernel_size)

    self.conv0_2 = DoubleConv(nb_filter[0], kernel_size)
    self.conv1_2 = DoubleConv(nb_filter[1], kernel_size)
    self.conv2_2 = DoubleConv(nb_filter[2], kernel_size)

    self.conv0_3 = DoubleConv(nb_filter[0], kernel_size)
    self.conv1_3 = DoubleConv(nb_filter[1], kernel_size)

    self.conv0_4 = DoubleConv(nb_filter[0], kernel_size)

    # All Upsampling convolution
    self.up1_0 = Conv2DTranspose(nb_filter[1], kernel_size=2, strides=2)
    self.up2_0 = Conv2DTranspose(nb_filter[2], kernel_size=2, strides=2)
    self.up3_0 = Conv2DTranspose(nb_filter[3], kernel_size=2, strides=2)
    self.up4_0 = Conv2DTranspose(nb_filter[4], kernel_size=2, strides=2)

    self.up1_1 = Conv2DTranspose(nb_filter[1], kernel_size=2, strides=2)
    self.up2_1 = Conv2DTranspose(nb_filter[2], kernel_size=2, strides=2)
    self.up3_1 = Conv2DTranspose(nb_filter[3], kernel_size=2, strides=2)

    self.up1_2 = Conv2DTranspose(nb_filter[1], kernel_size=2, strides=2)
    self.up2_2 = Conv2DTranspose(nb_filter[2], kernel_size=2, strides=2)
    
    self.up1_3 = Conv2DTranspose(nb_filter[1], kernel_size=2, strides=2)

    if(num_classes==1):
        self.final_conv= Conv2D(num_classes, 1, padding='same', activation=sigmoid)
    else:
        self.final_conv= Conv2D(num_classes, 1, padding='same', activation=softmax)


  def call(self,x):
    x0_0 = self.conv0_0(x)

    x1_0 = self.conv1_0(self.pool(x0_0))
    x0_1 = self.conv0_1(tf.concat([x0_0, self.up1_0(x1_0)], 3))

    x2_0 = self.conv2_0(self.pool(x1_0))
    x1_1 = self.conv1_1(tf.concat([x1_0, self.up2_0(x2_0)], 3))
    x0_2 = self.conv0_2(tf.concat([x0_0, x0_1, self.up1_1(x1_1)], 3))

    x3_0 = self.conv3_0(self.pool(x2_0))
    x2_1 = self.conv2_1(tf.concat([x2_0, self.up3_0(x3_0)], 3))
    x1_2 = self.conv1_2(tf.concat([x1_0, x1_1, self.up2_1(x2_1)], 3))
    x0_3 = self.conv0_3(tf.concat([x0_0, x0_1, x0_2, self.up1_2(x1_2)], 3))

    x4_0 = self.conv4_0(self.pool(x3_0))
    x3_1 = self.conv3_1(tf.concat([x3_0, self.up4_0(x4_0)], 3))
    x2_2 = self.conv2_2(tf.concat([x2_0, x2_1, self.up3_1(x3_1)], 3))
    x1_3 = self.conv1_3(tf.concat([x1_0, x1_1, x1_2, self.up2_2(x2_2)], 3))
    x0_4 = self.conv0_4(tf.concat([x0_0, x0_1, x0_2, x0_3, self.up1_3(x1_3)], 3))

    # self.final_conv(x0_1)
    # self.final_conv(x0_2)
    # self.final_conv(x0_3)
    return self.final_conv(x0_4)

  def get_summary(self, input_shape):
    inputs=Input(shape=input_shape)
    return Model(inputs=[inputs], outputs=[self.call(inputs)]).summary()
