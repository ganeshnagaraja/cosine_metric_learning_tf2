import tensorflow as tf


# class BasicBlock(tf.keras.layers.Layer):
#
#     def __init__(self, filter_num, stride=1):
#         super(BasicBlock, self).__init__()
#         self.conv1 = tf.keras.layers.Conv2D(filters=filter_num,
#                                             kernel_size=(3, 3),
#                                             strides=stride,
#                                             padding="same")
#         self.bn1 = tf.keras.layers.BatchNormalization()
#         self.conv2 = tf.keras.layers.Conv2D(filters=filter_num,
#                                             kernel_size=(3, 3),
#                                             strides=1,
#                                             padding="same")
#         self.bn2 = tf.keras.layers.BatchNormalization()
#         if stride != 1:
#             self.downsample = tf.keras.Sequential()
#             self.downsample.add(tf.keras.layers.Conv2D(filters=filter_num,
#                                                        kernel_size=(3, 3),
#                                                        strides=stride))
#             self.downsample.add(tf.keras.layers.BatchNormalization())
#         else:
#             self.downsample = lambda x: x
#
#     def call(self, inputs, training=None, **kwargs):
#         residual = self.downsample(inputs)
#
#         x = self.conv1(inputs)
#         x = self.bn1(x, training=training)
#         x = tf.nn.elu(x)
#         x = self.conv2(x)
#         x = self.bn2(x, training=training)
#         output = tf.nn.elu(tf.keras.layers.add([residual, x]))
#
#         return output
#
#
# def make_basic_block_layer(filter_num, blocks, stride=1):
#     res_block = tf.keras.Sequential()
#     res_block.add(BasicBlock(filter_num, stride=stride))
#
#     # for _ in range(1, blocks):
#     #     print(" yo boys -----------------")
#     #     res_block.add(BasicBlock(filter_num, stride=1))
#
#     return res_block


class Residual(tf.keras.Model):
    """The Residual block of ResNet."""
    def __init__(self, filter_num, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(
            filter_num, padding='same', kernel_size=(3, 3), strides=strides)
        self.bn1 = tf.keras.layers.BatchNormalization()

        self.conv2 = tf.keras.layers.Conv2D(
            filter_num, kernel_size=(3, 3), padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization()

        self.conv3 = None
        if use_1x1conv:
            self.conv3 = tf.keras.layers.Conv2D(
                filter_num, kernel_size=(1, 1), strides=strides)

    def call(self, inputs, **kwargs):
        output = tf.keras.activations.elu(self.bn1(self.conv1(inputs)))
        output = self.bn2(self.conv2(output))
        if self.conv3 is not None:
            inputs = self.conv3(inputs)
        output += inputs
        return tf.keras.activations.elu(output)

# class ResnetBlock(tf.keras.layers.Layer):
#     def __init__(self, filter_num, num_residuals, first_block=False,
#                  **kwargs):
#         super(ResnetBlock, self).__init__(**kwargs)
#         self.residual_layers = []
#         for i in range(num_residuals):
#             if i == 0 and not first_block:
#                 self.residual_layers.append(
#                     Residual(filter_num, use_1x1conv=True, strides=2))
#             else:
#                 self.residual_layers.append(Residual(filter_num))
#
#     def call(self, X):
#         for layer in self.residual_layers.layers:
#             X = layer(X)
#         return X

# self.net = tf.keras.Sequential([
        #     tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=1, padding="same"),
        #     tf.keras.layers.BatchNormalization(),
        #     tf.keras.layers.Activation("elu"),
        #     tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=1, padding="same"),
        #     tf.keras.layers.BatchNormalization(),
        #     tf.keras.layers.Activation("elu"),
        #     tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=2, padding="same")
        # ])
