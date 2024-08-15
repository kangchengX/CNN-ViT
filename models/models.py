import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.models as models
from typing import List, Callable, Tuple
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG16, VGG19
from tensorflow.keras.applications import ResNet50, ResNet101
from typing import Literal

class MlpBlock(Layer):
    """2-layer mlp block implementaion."""

    def __init__(
            self, 
            dim: int, 
            hidden_dim: int, 
            dropout: float | None = 0.5
    ):
        """
        Initialize the model.
        
        Args:
            dim (int): dimension of the inputs and outputs, i.e. dimension of the words vector.
            hidden_dim (int): hidden dimension.
            dropout (float): dropout percentage. Default to `0.5`.
        """
        
        super().__init__()
        self.net = models.Sequential([
            layers.Dense(hidden_dim, activation=tf.nn.gelu),
            layers.Dropout(dropout),
            layers.Dense(dim),
            layers.Dropout(dropout),
        ])

    def call(self, inputs):
        return self.net(inputs)


class TransformerBlock(Layer):
    """Transformer block implementation."""

    def __init__(
            self, 
            dim: int, 
            num_heads: int, 
            mlp_dim: int, 
            dropout: float | None = 0.5
    ):
        """
        Initialize the model.
          
        Args:
            dim (int): dimension of the words vector.
            num_heads (int): number of heads.
            mlp_dim (int): hidden dimension of mlp blocks.
            dropout (float): dropout percentage. Default to `0.5`.
        """
        
        super().__init__()
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=dim)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.mlp = MlpBlock(dim, hidden_dim=mlp_dim, dropout=dropout)

    def call(self, inputs):
        # first residual connection flow
        outputs1 = self.norm1(inputs)
        outputs1 = self.attention(query=outputs1, key=outputs1, value=outputs1)
        outputs1 = inputs + outputs1
        
        # second residual connection flow
        outputs2 = self.norm2(outputs1)
        outputs2 = self.mlp(outputs2)
        outputs = outputs1 + outputs2

        return outputs


class ViT(Model):
    """Visiton trandormer implementation."""

    def __init__(
            self, 
            image_size: int | Tuple[int, int, int], 
            patch_size: int | Tuple[int, int], 
            num_classes: int, 
            dim: int, 
            depth: int, 
            num_heads: int, 
            mlp_dim: int, 
            dropout: float | None = 0.5
    ):
        """
        Initialize the model.
        
        Args:
            image_size (int | tuple): image_size = image_height = image_width and channels will be inferred as 3 if int, (image_heigh, image_width, channels) if tuple.
            patch_size (int | tuple): patch_size = patch_height = path_width if int, (patch_heigh, patch_width) if tuple.
            num_classes (int): number of the classes.
            dim (int): dimension of the words vector.
            depth (int): number of transformer blocks.
            heads (int): number of heads.
            mlp_dim (int): dimension of feddforward blocks.
            dropout (float): dropout percentage. Default to `0.5`.
        """
        
        super().__init__()
        if isinstance(image_size, int):
            image_size = (image_size, image_size, 3)
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        # sizes and shapes
        self.patch_size = patch_size
        self.dim = dim
        self.patch_dim = patch_size[0] * patch_size[1] * image_size[2]

        num_patches = image_size[0] // patch_size[0] * image_size[1] // patch_size[1]

        # embedding
        self.pos_embedding = self.add_weight(
            name="position_embeddings", 
            shape=(1, num_patches + 1, dim), 
            initializer=tf.random_normal_initializer()
        )

        self.cls_token = self.add_weight(
            name="cls_token", 
            shape=(1, 1, dim), 
            initializer=tf.random_normal_initializer()
        )
        
        # initial layers/blocks
        self.patch_proj = layers.Dense(dim)
        self.transformer_blocks = models.Sequential(
            [TransformerBlock(dim, num_heads, mlp_dim, dropout) for _ in range(depth)]
        )
        self.to_cls_token = layers.Identity()
        self.mlp_head = models.Sequential([
            layers.LayerNormalization(epsilon=1e-6),
            layers.Dense(mlp_dim, activation=tf.nn.gelu),
            layers.Dropout(dropout),
            layers.Dense(num_classes),
        ])

    def call(self, images, training = False):
        shapes = tf.shape(images)
        batch_size, _, _, _ = tf.unstack(shapes)

        #image to flattened patches
        outputs = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size[0], self.patch_size[1], 1],
            strides=[1, self.patch_size[0], self.patch_size[1], 1],
            rates=[1, 1, 1, 1],
            padding="VALID"
        )
        outputs = tf.reshape(outputs, [batch_size, -1, self.patch_dim])

        # flattened pathes to word vectors
        outputs = self.patch_proj(outputs)

        # cls token and position embedding
        cls_tokens = tf.broadcast_to(self.cls_token, [batch_size, 1, self.dim])
        outputs = tf.concat([cls_tokens, outputs], axis=1)
        outputs += self.pos_embedding
        
        # feed word vectors to the network
        outputs = self.transformer_blocks(outputs,training=training)
        outputs = self.to_cls_token(outputs[:, 0])
        outputs =  self.mlp_head(outputs, training=training)

        return outputs
    
class VGG(Model):
    """VGG model."""

    def __init__(
            self,
            config_arch: Literal['vgg16', 'vgg19'],
            image_size: int | Tuple[int, int, int],
            num_classes: int,
            dropout: float | None = 0.5
    ):
        """
        Initialize the model.

        Args:
            config_arch (str): config architecture. Must be `'vgg16'` or `'vgg19'`.
            image_size (int | tuple): image_size = image_height = image_width and channels will be inferred as 3 if int, (image_heigh, image_width, channels) if tuple.
            num_classes (int): number of the classes.
            dropout (float): dropout percentage. Default to `0.5`.
        """
        super().__init__()
        if isinstance(image_size, int):
            image_size = (image_size, image_size, 3)
        if config_arch == 'vgg16':
            self.conv_block = VGG16(include_top=False, weights=None, input_shape=image_size, pooling='max')
        elif config_arch == 'vgg19':
            self.conv_block = VGG19(include_top=False, weights=None, input_shape=image_size, pooling='max')
        else:
            raise ValueError(f"Unsupported config_arch {config_arch}")

        self.mlp = models.Sequential([
            layers.Dense(4096, activation='relu'),
            layers.Dropout(dropout),
            layers.Dense(4096, activation='relu'),
            layers.Dropout(dropout),
            layers.Dense(num_classes, activation='softmax')
        ])

    def call(self, inputs):
        conv_outputs = self.conv_block(inputs)
        outputs = self.mlp(conv_outputs)

        return outputs

class ResNet(Model):
    """ResNet."""
    def __init__(
            self,
            config_arch: Literal['resnet50', 'resnet101'],
            image_size: int | Tuple[int, int, int],
            num_classes: int
    ):
        """
        Initialize the model.

        Args:
            config_arch (str): config architecture. Must be `'resnet50'` or `'resnet101'`.
            image_size (int | tuple): image_size = image_height = image_width and channels will be inferred as 3 if int, (image_heigh, image_width, channels) if tuple.
            num_classes (int): number of the classes.
            dropout (float): dropout percentage. Default to `0.5`.
        """
        super().__init__()
        if isinstance(image_size, int):
            image_size = (image_size, image_size, 3)
        if config_arch == 'resnet50':
            self.conv_block = ResNet50(include_top=False, weights=None, input_shape=image_size, pooling='avg')
        elif config_arch == 'resnet101':
            self.conv_block = ResNet101(include_top=False, weights=None, input_shape=image_size, pooling='avg')
        else:
            raise ValueError(f"Unsupported config_arch {config_arch}")

        self.classifier = layers.Dense(num_classes, activation='softmax')

    def call(self, inputs):
        conv_outputs = self.conv_block(inputs)
        outputs = self.classifier(conv_outputs)

        return outputs
    