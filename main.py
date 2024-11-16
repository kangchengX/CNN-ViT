import argparse, os, json, warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Filter out INFO and WARNING messages.
import tensorflow as tf
import pandas as pd
import numpy as np
from typing import Literal
from config import get_mobilevit_config
from data import DataLoader
from models.models import VGG, ResNet, ViT
from models.mobileViT import MobileViT
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, balanced_accuracy_score
tf.get_logger().setLevel('ERROR') # Filter out "WARNING:tensorflow:"


def create_model(
    config_arch: Literal['mobilevit_xxs', 'mobilevit_xs', 'mobilevit_s', 'vgg16', 'vgg19', 'resnet50', 'resnet101', 'vit'],
    image_size: int,
    image_channels: int,
    num_classes: int,
    dropout: float,
    vit_patch_size: int | None = None,
    vit_dim: int | None = None,
    vit_depth: int | None = None,
    vit_num_heads: int | None = None,
    vit_mlp_dim: int | None = None,
):
    """
    Create model.

    Args:
        config_arch (str): architecture of the model to create.
        image_size (int): image_size = image_height = image_width.
        image_channels (int): channels of the input image.
        num_classes (int): number of classes.
        dropout (float): dropout rate.
        vit_patch_size (int): path size of the vit model.
        vit_dim (int): word dimension in the vit model.
        vit_depth (int): number of transformer blocks in the vit model.
        vit_num_heads (int): number of heads in the vit model.
        vit_mlp_dim (int): dimension of the mlp hidden layer in the vit model.
    
    Returns:
        model (MobileViT | VGG | ResNet | ViT): tensorflow model.
    """

    if config_arch in ['mobilevit_xxs', 'mobilevit_xs', 'mobilevit_s']:
        config = get_mobilevit_config(
            config_arch=config_arch, 
            num_classes=num_classes, 
            image_size=image_size, 
            image_channels=image_channels,
            dropout=dropout
        )
        model = MobileViT(**config)
    elif config_arch in ['vgg16', 'vgg19']:
        model = VGG(
            config_arch=config_arch, 
            image_size=(image_size, image_size, image_channels),
            num_classes=num_classes,
            dropout=dropout
        )
    elif config_arch in ['resnet50', 'resnet101']:
        model = ResNet(
            config_arch=config_arch, 
            image_size=(image_size, image_size, image_channels),
            num_classes=num_classes
        )
    elif config_arch == 'vit':
        model = ViT(
            image_size=(image_size, image_size, image_channels),
            patch_size=vit_patch_size,
            num_classes=num_classes,
            dim=vit_dim,
            depth=vit_depth,
            num_heads=vit_num_heads,
            mlp_dim=vit_mlp_dim,
            dropout=dropout
        )
    else:
        raise ValueError(f'Unsupported config_arch {config_arch}')

    return model

class Encoder(json.JSONEncoder):
    """Encoder for json dump. Focusing on numpy type conversion."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return super().default(obj)
        
if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # model structure related
    parser.add_argument('config_arch', type=str, help='Architecture of the model. \
                        Value shoud be mobilevit_xxs, mobilevit_xs, mobilevit_s, vgg16, vgg19, resnet50, resnet101 or vit.')
    parser.add_argument('--image_size', type=int, default=224, help='Height or width of the input image. Default to 224.')
    parser.add_argument('--image_channels', type=int, default=3, help='Channels of the input image. Default to 3.')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate for the mlp layers in these models. Default to 0.5.')
    parser.add_argument("--vit_patch_size", type=int, default=2, help='Patch size for the Vision Transformer. Default to 2.')
    parser.add_argument("--vit_dim", type=int, default=768, help='(Word) dimension of the Vision Transformer. Default to 768.')
    parser.add_argument("--vit_depth", type=int, help='Number of layers in the Vision Transformer.')
    parser.add_argument("--vit_num_heads", type=int, help='Number of attention heads in the Vision Transformer.')
    parser.add_argument("--vit_mlp_dim", type=int, default=1536, help='Dimension of the mlp hidden layer in the Vision Transformer.')
    
    # data related
    parser.add_argument('--split_ratio', type=float, default=0.75, help='Ratio of the training set in the whole dataset.')
    parser.add_argument('--data_folder', type=str, default='data', help='Folder containing the data.')
    parser.add_argument('--not_shuffle', action='store_false', help='Not to shuffle the dataset.')

    # training related
    parser.add_argument('--num_epochs', type=int, default=20, help='Number of epochs. Default to 20.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size. Default to 16.')
    parser.add_argument('--learning_rate', type=float, default=1e-2, help='Learning rate. Default to 0.01.')

    # results saving related
    parser.add_argument('--results_filename', type=str, default='results', help='Path to save the results. \
                        The extension in the input string will be ignored, and is automatically managed. \
                        If the model is a vit model, the extension is .csv. If not, the extension is .json.')

    args = parser.parse_args()

    # load results file
    root, _ = os.path.splitext(args.results_filename)
    file_path = root + '.csv' if args.config_arch == 'vit' else root + '.json'
    if os.path.exists(file_path):
        if args.config_arch == 'vit':
            results = pd.read_csv(file_path)
        else:
            with open(file_path, 'r') as f:
                results = json.load(f)
    else:
        results = pd.DataFrame(columns=[
                'word_dim', 
                'depth', 
                'num_heads', 
                'learning_rate',
                'batch_size',
                'num_trainable_params', 
                'last_train_loss', 
                'last_train_accuracy', 
                'test_loss', 
                'test_accuracy'
            ]
        ) if args.config_arch == 'vit' else {}

    # load data
    data_loader = DataLoader(args.image_size, ratio=args.split_ratio)
    data_loader.load(args.data_folder, shuffle=args.not_shuffle)
    
    # model training
    model = create_model(
        args.config_arch, 
        image_size=args.image_size, 
        image_channels=args.image_channels, 
        num_classes=len(data_loader.classes),
        dropout=args.dropout,
        vit_patch_size=args.vit_patch_size,
        vit_dim=args.vit_dim,
        vit_depth=args.vit_depth,
        vit_num_heads=args.vit_num_heads,
        vit_mlp_dim=args.vit_mlp_dim
    )

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    print('-'*40 + 'train' + '-'*40)
    history = model.fit(data_loader.images_train, data_loader.labels_train, epochs=args.num_epochs, batch_size=args.batch_size)

    train_accuracies = history.history['accuracy']
    train_losses = history.history['loss']

    # model test
    print('-'*40 + 'test' + '-'*40)
    y_pred_probs = model.predict(data_loader.images_test)
    y_pred = tf.argmax(y_pred_probs, axis=1).numpy()
    y_true = data_loader.labels_test

    test_loss = tf.keras.losses.SparseCategoricalCrossentropy()(y_true, y_pred_probs).numpy()
    test_accuracy = accuracy_score(y_true, y_pred)
    test_weighted_accuracy = balanced_accuracy_score(y_true, y_pred)
    test_f1_score = f1_score(y_true, y_pred, average='macro')
    test_confidence_score = np.mean(np.max(y_pred_probs, axis=1))
    test_confusion_matrix = confusion_matrix(y_true, y_pred)

    num_trainable_params = sum([tf.size(v).numpy() for v in model.trainable_weights])

    # save results
    if args.config_arch == 'vit':
        results.loc[len(results)] = [args.vit_dim, args.vit_depth, args.vit_num_heads, args.learning_rate, args.batch_size, num_trainable_params, train_losses[-1], train_accuracies[-1], test_loss, test_accuracy]
        results.to_csv(file_path, index=False)
    else:
        if args.config_arch in results:
            warnings.warn(f"The model architecture {args.config_arch} has been examined before. The results will be overwritten.")
        results[args.config_arch] = {
            'num_trainable_params' : num_trainable_params, 
            'train_losses' : train_losses, 
            'train_accuracies' : train_accuracies, 
            'test_loss' : test_loss, 
            'test_accuracy' : test_accuracy,
            'test_weighted_accuracy' : test_weighted_accuracy,
            'test_f1_score' : test_f1_score,
            'test_confidence_score' : test_confidence_score,
            'test_confusion_matrix' : test_confusion_matrix.tolist()
        }
        with open(file_path, 'w') as f:
            json.dump(results, f, cls=Encoder, indent=4)
    