import tensorflow as tf
from residual_block import Residual

NUM_CLASSES = 184
image_height = 128
image_width = 64
channels = 3


class SiameseNet(tf.keras.Model):
    def __init__(self, classes):
        super(SiameseNet, self).__init__()

        self.conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=1, padding="same", input_shape=(128, 64, 3))
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.act1 = tf.keras.layers.Activation("elu")

        self.conv2 = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=1, padding="same", input_shape=(128, 64, 3))
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.act2 = tf.keras.layers.Activation("elu")

        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=2, padding="same")

        # # adding the residual blocks
        self.layer1 = Residual(filter_num=32, use_1x1conv=True, strides=1)
        self.layer2 = Residual(filter_num=32, use_1x1conv=True, strides=1)
        self.layer3 = Residual(filter_num=64, use_1x1conv=True, strides=2)
        self.layer4 = Residual(filter_num=64, use_1x1conv=True, strides=1)
        self.layer5 = Residual(filter_num=128, use_1x1conv=True, strides=2)
        self.layer6 = Residual(filter_num=128, use_1x1conv=True, strides=1)

        # flatten before the dense layer
        self.flatten = tf.keras.layers.Flatten()

        # Dense layer
        self.fc = tf.keras.layers.Dense(units=NUM_CLASSES, activation=tf.keras.activations.elu)

    def call_once(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.act1(x)
        x = self.bn1(x, training=training)
        x = self.conv2(inputs)
        x = self.act2(x)
        x = self.bn2(x, training=training)
        x = self.pool1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.flatten(x)
        x = self.fc(x)
        output = tf.nn.l2_normalize(x)
        return output

    def call(self, input1, input2 ,input3=None):
        output1 = self.call_once(input1)
        output2 = self.call_once(input2)

        if input3 is not None:
            output3 = self.call_once(input3)
            return output1,output2,output3

        return output1, output2


class TripletLoss(tf.keras.Model):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def call(self, anchor, positive, negative, size_average=True):
        distance_positive = tf.keras.losses.cosine_similarity(anchor,positive) #Each is batch X 512
        distance_negative = tf.keras.losses.cosine_similarity(anchor,negative)  # .pow(.5)
        losses = (1- distance_positive)**2 + (0 - distance_negative)**2
        print(losses)#Margin not used in cosine case.
        return tf.math.reduce_mean(losses) if size_average else tf.reduce_sum(losses)


# y = tf.ones((1, 128, 64, 3))
# x = tf.fill([1, 128, 64, 3], 2.0)
# z = tf.fill([1, 128, 64, 3], 3.0)
# model = SiameseNet(NUM_CLASSES)
# output_1, output_2, output_3  = model.call(x, y, z)
#
#
# anchor = tf.fill([1, 128, 64, 3], 81.0)
# criterion = TripletLoss(margin=1)
# triplet_loss = criterion(output_1, output_2, output_3)
# print(triplet_loss)