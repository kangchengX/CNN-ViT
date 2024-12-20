import math
import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.models as models
from typing import Tuple, Literal
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model
from models.models import TransformerBlock


class ConvNormActivate(Layer):
    """
    2D convolutional layer with batch normalization and swish activation.
    """
    def __init__(self, filters: int, kernel_size: int, strides: int, padding: Literal['same', 'valid'] | None = "same"):
        """
        Initialize the model.

        Args:
            filters (int): number of filters in convolution.
            kernel_size (int): size of the conv kernel.
            strides (int): stride in convolution.
            padding (str): padding strategy. Default to `"same"`.
        """
        super().__init__()
        self.conv_layer = layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, use_bias=False)
        self.bn = layers.BatchNormalization()

    def call(self, inputs):
        outputs = self.conv_layer(inputs)
        outputs = self.bn(outputs)
        outputs = tf.nn.swish(outputs)

        return outputs


class InvertedResidualBlock(Layer):
    """
    The Inverted Residual Block.

    conv 1x1 (expand channels) -> batch norm -> swish
    -> depth-wise conv -> batch norm -> swish
    -> conv 1x1 (reduce channels) -> batch norm
    -> inputs + outputs if `input_channels` == `output_channels` and `strides` == 1, just outputs otherwise
    """
    def __init__(self, input_channels: int, expanded_channels: int, output_channels: int, strides: int):
        """
        Initialize the layer.

        Args:
            input_channels (int): channels of the input tensor.
            expanded_channels (int): channels of the output tensor of the first 1x1 conv.
            output_channels (int): channels of the output tensor of this layer.
            strides (int): strides of the depth wise conv.
        """
        super().__init__()        
        self.expand_conv = layers.Conv2D(expanded_channels, kernel_size=1, strides=1, padding="same", use_bias=False)
        self.expand_bn = layers.BatchNormalization()

        self.depthwise_conv = layers.DepthwiseConv2D(3, strides=strides, padding="same", use_bias=False)
        self.depthwise_bn = layers.BatchNormalization()
        self.project_conv = layers.Conv2D(output_channels, kernel_size=1, strides=1, padding="same", use_bias=False)
        self.project_bn = layers.BatchNormalization()

        if input_channels == output_channels and strides == 1:
            self.add = layers.Add()
        else:
            self.add = layers.Lambda(lambda tensors: tensors[1])
        
    def call(self, inputs):
        # expand channels
        outputs = self.expand_conv(inputs)
        outputs = self.expand_bn(outputs)
        outputs = tf.nn.swish(outputs)

        # depthsize conv
        outputs = self.depthwise_conv(outputs)
        outputs = self.depthwise_bn(outputs)
        outputs = tf.nn.swish(outputs)

        # reduce channels
        outputs = self.project_conv(outputs)
        outputs = self.project_bn(outputs)

        # residual connection if `input_channels` == `output_channels` and `strides` == 1
        outputs = self.add([inputs, outputs])

        return outputs


class GroupTensor(Layer):
    """
    Prepare tensor for grouped tensformer.

    (b, (h_num x hp), (w_num x wp), c) -> ((b x hp x wp), (h_num x w_num), c)
    """
    def __init__(
        self, 
        image_size: int | Tuple[int, int, int], 
        patch_size: int | tuple[int, int], 
        channels: int
    ):
        """
        Initialize the layer.

        Args:
            image_size (int | tuple): image_size = image_height = image_width and channels will be inferred as 3 if int, (image_heigh, image_width, channels) if tuple.
            patch_size (int | tuple): patch_size = patch_height = path_width if int, (patch_heigh, patch_width) if tuple.
            channels (int): channels of the input tensor.
        """
        super().__init__()

        if isinstance(image_size, int):
            image_size = (image_size, image_size, 3)
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)

        image_height_new = math.ceil(image_size[0] / patch_size[0]) * patch_size[0]
        image_width_new = math.ceil(image_size[1] / patch_size[1]) * patch_size[1]

        self.image_height_new = image_height_new
        self.image_width_new = image_width_new
        self.channels = channels
        self.patch_size = patch_size

        # Layer to ensure the the image size is divisable by the patch size
        if self.image_height_new != image_size[0] or self.image_width_new != image_size[1]:
            self.resize = layers.Resizing(height = self.image_height_new, width = self.image_width_new)
        else:
            self.resize = layers.Identity()
        
    def call(self, inputs):
        # resize image if needed so that image_size can be divisible by patch_size
        outputs = self.resize(inputs)

        # reshape tensor to grouped tensors, i.e., unfold the tensor
        # (b, (h_num x hp), (w_num x wp), c) -> (b, h_num, hp, w_num, wp, c)
        outputs = tf.reshape(
            outputs, 
            (
                -1,
                self.image_height_new // self.patch_size[0],
                self.patch_size[0],
                self.image_width_new // self.patch_size[1],
                self.patch_size[1],
                self.channels
            )
        )
        # (b, h_num, hp, w_num, wp, c) -> (b, hp, wp, h_num, w_num, c)
        outputs = tf.transpose(outputs, (0, 2, 4, 1, 3, 5))
        # (b, hp, wp, h_num, w_num, c) -> (b x hp x wp), (hi x wi), c)
        outputs = tf.reshape(outputs, (
            -1, 
            self.image_height_new // self.patch_size[0] * self.image_width_new // self.patch_size[1], 
            self.channels
        ))

        return outputs


class ReshapeGroupedTensor(Layer):
    """
    Reshape tensor after the grouped tensformer.

    ((b x hp x wp), (h_num x w_num), c) -> (b, (h_num x hp), (w_num x wp), c)

    (b, (hp, wp), (hi, wi), c) -> (b, (hi, hp), (wi, wp), c)
    """
    def __init__(
        self, 
        image_size: int | Tuple[int, int, int], 
        patch_size: int | tuple[int, int], 
        channels: int
    ):
        """
        Initialize the layer.

        Args:
            image_size (int | tuple): image_size = image_height = image_width and channels will be inferred as 3 if int, (image_heigh, image_width, channels) if tuple.
            patch_size (int | tuple): patch_size = patch_height = path_width if int, (patch_heigh, patch_width) if tuple.
            channels (int): channels of the input tensor.
        """
        super().__init__()

        if isinstance(image_size, int):
            image_size = (image_size, image_size, 3)
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)

        image_height_new = math.ceil(image_size[0] / patch_size[0]) * patch_size[0]
        image_width_new = math.ceil(image_size[1] / patch_size[1]) * patch_size[1]

        self.image_height_new = image_height_new
        self.image_width_new = image_width_new
        self.channels = channels
        self.patch_size = patch_size

        # Layer to convert the image to the original size
        if self.image_height_new != image_size[0] or self.image_width_new != image_size[1]:
            self.resize = layers.Resizing(height = image_size[0], width = image_size[1])
        else:
            self.resize = layers.Identity()
        
    def call(self, inputs):
        # fold the tensor       
        # (b x hp x wp), (h_num x w_num), c) -> (b, hp, wp, h_num, w_num, c)
        outputs = tf.reshape(inputs, (
            -1,
            self.patch_size[0],
            self.patch_size[1],
            self.image_height_new // self.patch_size[0],
            self.image_width_new // self.patch_size[1],
            self.channels
        ))
        # (b, hp, wp, h_num, w_num, c) -> (b, h_num, hp, w_num, wp, c)
        outputs = tf.transpose(outputs, (0, 3, 1, 4, 2, 5))
        # (b, h_num, hp, w_num, wp, c) -> (b, (h_num x hp), (w_num x wp), c)
        outputs = tf.reshape(outputs, (
            -1, 
            self.image_height_new,
            self.image_width_new,
            self.channels
        ))

        # image size conversion
        outputs = self.resize(outputs)

        return outputs
    

class MobileViTBlock(Layer):
    """
    The mobile vit block.

    conv 3x3 -> conv 1x1 -> grouped transformer for multiple times-> conv 1x1 -> concat -> conv 3x3
    """
    def __init__(
        self, 
        num_transformer_blocks: int, 
        input_channels: int,
        projection_dim: int, 
        num_heads: int,
        mlp_dim: int,
        image_size: int | Tuple[int, int, int],
        patch_size: int | Tuple[int, int] | None = 2,
        dropout: float | None = 0.5
    ):
        """
        Initialize the layer.

        Args:
            num_transformer_blocks (int): number of the transformer blocks.
            input_channels (int): input channels of the tensor.
            projection_dim (int): 'word' diemention of  the tensor sent to transformer.
            num_heads (int): number of heads in the transformer.
            mlp_dim: dimension of mlp layer in the transformer.
            image_size (int | tuple): image_size = image_height = image_width and channels will be inferred as 3 if int, (image_heigh, image_width, channels) if tuple.
            patch_size (int | tuple): patch_size = patch_height = path_width if int, (patch_heigh, patch_width) if tuple.
            dropout (float): dropout in the mlp layer.
        """
        super().__init__()
        self.local_conv1 = ConvNormActivate(filters=input_channels, kernel_size=3, strides=1, padding="same")
        self.local_conv2 = ConvNormActivate(filters=projection_dim, kernel_size=1, strides=1, padding="same")
        self.group_tensor = GroupTensor(image_size=image_size, patch_size=patch_size, channels=projection_dim)
        
        self.transformers = models.Sequential()
        for _ in range(num_transformer_blocks):
            self.transformers.add(TransformerBlock(dim=projection_dim, num_heads=num_heads, mlp_dim=mlp_dim, dropout=dropout, activation=tf.nn.swish))

        self.reshape_grouped_tensor = ReshapeGroupedTensor(image_size=image_size, patch_size=patch_size, channels=projection_dim)

        self.local_conv3 = ConvNormActivate(filters=input_channels, kernel_size=1, strides=1, padding="same")
        self.concat_tensors = layers.Concatenate(axis=-1)
        self.local_conv4  = ConvNormActivate(filters=input_channels, kernel_size=3, strides=1, padding="same")
    
    def call(self, inputs):
        # convolution
        local_features = self.local_conv1(inputs)
        local_features = self.local_conv2(local_features)

        # transformer by groups
        grouped_features = self.group_tensor(local_features)
        grouped_transformed_features = self.transformers(grouped_features)
        transformed_features = self.reshape_grouped_tensor(grouped_transformed_features)

        # convolution
        outputs = self.local_conv3(transformed_features)
        outputs = self.concat_tensors((inputs, outputs))
        outputs = self.local_conv4(outputs)

        return outputs
    

class MobileViT(Model):
    """
    The MobileVit model. See https://arxiv.org/abs/2110.02178.
    """
    def __init__(
        self, 
        channels: list,
        dims: list,
        mlp_dims: list,
        num_classes: int, 
        image_size: int,
        expansion_factor: int | None = 2,
        patch_size: int | None = 2,
        dropout: float | None = 0.5,
        last_conv_expansion_factor: int | None = 4
    ):
        """
        Initialize the model.

        Args:
            channels (list): list of 8 channels of the inverted resudual block. 
                The input channels and output channels for inverted resudual block i (start at 1) are `channels[i-1]` and `channels[i]`.
            dims (list): list of 3 dims of the 'word vector' in the transformer. 
                The dim of the 'word vector' in the transformer for MobileViTBlock i (start at 1) is `dims[i-1]`.
            mlp_dims: list of dimension of the correspong mlp dim in transformer.
            num_classes (int): number of classes.
            image_size (int): image_size = image_height = image_width.
            expansion_factor: expansion_factor in the inverted resudual block.
            patch_size (int): patch_size = patch_height = path_width.
            dropout (float): dropout rate in the mlp.
            last_conv_expansion_factor (int): expansion factor for channels in the last convolutional layer.
        """
        super().__init__()
        self.conv3x3 = ConvNormActivate(filters=channels[0], kernel_size=3, strides=2, padding="same")
        self.mv2_block1 = InvertedResidualBlock(channels[0], channels[0] * expansion_factor, channels[1], 1)
        self.mv2_block2 = InvertedResidualBlock(channels[1], channels[1] * expansion_factor, channels[2], 2)
        self.mv2_block3 = InvertedResidualBlock(channels[2], channels[2] * expansion_factor, channels[3], 1)
        self.mv2_block4 = InvertedResidualBlock(channels[3], channels[3] * expansion_factor, channels[4], 1)
        self.mv2_block5 = InvertedResidualBlock(channels[4], channels[4] * expansion_factor, channels[5], 2)
        self.mobilevit_block1 = MobileViTBlock(
            num_transformer_blocks=2, 
            input_channels=channels[5],
            projection_dim=dims[0], 
            num_heads=2, 
            mlp_dim=mlp_dims[0],
            image_size=image_size // 8,
            patch_size=patch_size,
            dropout=dropout
        )
        self.mv2_block6 = InvertedResidualBlock(channels[5], channels[5] * expansion_factor, channels[6], 2)
        self.mobilevit_block2 = MobileViTBlock(
            num_transformer_blocks=4, 
            input_channels=channels[6],
            projection_dim=dims[1], 
            num_heads=2,
            mlp_dim=mlp_dims[1],
            image_size=image_size // 16,
            patch_size=patch_size,
            dropout=dropout
        )
        self.mv2_block7 = InvertedResidualBlock(channels[6], channels[6] * expansion_factor, channels[7], 2)
        self.mobilevit_block3 = MobileViTBlock(
            num_transformer_blocks=3, 
            input_channels=channels[7],
            projection_dim=dims[2], 
            num_heads=2,
            mlp_dim=mlp_dims[2],
            image_size=image_size // 32,
            patch_size=patch_size,
            dropout=dropout
        )

        self.conv1x1 = ConvNormActivate(filters=last_conv_expansion_factor * channels[7], kernel_size=1, strides=1, padding="same")
        self.global_avg_pool = layers.GlobalAvgPool2D()
        self.classifier = layers.Dense(num_classes, activation="softmax")

    def call(self, inputs):
        # layer 1
        outputs = self.conv3x3(inputs)
        outputs = self.mv2_block1(outputs)
        # layer 2
        outputs = self.mv2_block2(outputs)
        outputs = self.mv2_block3(outputs)
        outputs = self.mv2_block4(outputs)
        # layer 3
        outputs = self.mv2_block5(outputs)
        outputs = self.mobilevit_block1(outputs)
        # layer 4
        outputs = self.mv2_block6(outputs)
        outputs = self.mobilevit_block2(outputs)
        # layer 5
        outputs = self.mv2_block7(outputs)
        outputs = self.mobilevit_block3(outputs)
        # out
        outputs = self.conv1x1(outputs)
        outputs = self.global_avg_pool(outputs)
        outputs = self.classifier(outputs)
        return outputs
    