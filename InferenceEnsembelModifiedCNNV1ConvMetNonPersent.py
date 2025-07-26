import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, f1_score, matthews_corrcoef
from datetime import datetime
import models
import utils
import subprocess
from sklearn.model_selection import train_test_split
import cv2
from keras import backend as K 
import re
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
K.clear_session()
try:
    subprocess.check_output('nvidia-smi')
    print('Nvidia GPU detected!')
    print(tf.config.list_physical_devices('GPU'))
except Exception: # this command not being found can raise quite a few different errors depending on the configuration
    print('No Nvidia GPU in system!')

import argparse
from tensorflow.keras.applications import (
    ResNet50, ResNet101, ResNet152V2, InceptionResNetV2, MobileNet, MobileNetV2, NASNetMobile, VGG19,
    ResNet50V2, ResNet101V2
)
from tensorflow.keras.applications import efficientnet_v2
from tensorflow.keras.applications import EfficientNetV2B0, EfficientNetV2B3, EfficientNetV2S, EfficientNetV2M, EfficientNetV2L
# === MAPPING PREPROCESS ===
from tensorflow.keras.applications.vgg19 import preprocess_input as preprocess_vgg19
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_resnet50
from tensorflow.keras.applications.resnet_v2 import preprocess_input as preprocess_resnet50v2
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input as preprocess_inceptionresnetv2
from tensorflow.keras.applications.mobilenet import preprocess_input as preprocess_mobilenet
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as preprocess_mobilenetv2
from tensorflow.keras.applications.nasnet import preprocess_input as preprocess_nasnetmobile
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input as preprocess_efficientnetv2

# === MAPPING MODEL TYPE TO PREPROCESS FUNCTION ===
model_preprocessors = {
    "VGG19": preprocess_vgg19,
    "ResNet50": preprocess_resnet50,
    "ResNet101": preprocess_resnet50,
    "ResNet152V2": preprocess_resnet50v2,
    "InceptionResNetV2": preprocess_inceptionresnetv2,
    "MobileNet": preprocess_mobilenet,
    "MobileNetV2": preprocess_mobilenetv2,
    "NASNetMobile": preprocess_nasnetmobile,
    "ResNet50V2": preprocess_resnet50v2,
    "ResNet101V2": preprocess_resnet50v2,
    "EfficientNetV2B0": preprocess_efficientnetv2,
    "EfficientNetV2B3": preprocess_efficientnetv2,
    "EfficientNetV2S": preprocess_efficientnetv2,
    "EfficientNetV2M": preprocess_efficientnetv2,
    "EfficientNetV2L": preprocess_efficientnetv2,
}

    # === Daftar model yang ingin digunakan dalam ensemble ===

def repeatChannel(x, channel):
    # Reshape the images to 3D arrays with a grayscale channel
    x = np.expand_dims(x, axis=-1)    

    # Repeat the grayscale channel three times to create 3-channel images
    x = np.repeat(x, channel, axis=-1)
    return x

def apply_preprocess_input(X_train, X_val, X_test, model_type):
    """
    Apply model-specific preprocess_input to train, val, test sets.
    """
    preprocess_fn = model_preprocessors.get(model_type)
    if preprocess_fn is None:
        raise ValueError(f"Model type '{model_type}' belum ada di mapping preprocess_input.")
    
    X_train = preprocess_fn(X_train)
    X_val = preprocess_fn(X_val)
    X_test = preprocess_fn(X_test)

    return X_train, X_val, X_test

# === LOAD DATA ===
def load_npz_train_test_split_custom(pathTrain, pathTest, sizeImage, sizeTest, resize_meth):
    '''
    Load and resize dataset, no preprocessing here.
    '''
    dataset = np.load(pathTrain)
    X_train = dataset['data']
    y_train = dataset['label_code']

    dataset = np.load(pathTest)
    X_test = dataset['data']
    y_test = dataset['label_code']
    y_test_label = dataset['label']

    # Resize
    if resize_meth == "nearest":
        X_train_resized = np.array([cv2.resize(x, (sizeImage, sizeImage), interpolation=cv2.INTER_NEAREST) for x in X_train])
        X_test_resized = np.array([cv2.resize(x, (sizeImage, sizeImage), interpolation=cv2.INTER_NEAREST) for x in X_test])
    elif resize_meth == "bicubic":
        X_train_resized = np.array([cv2.resize(x, (sizeImage, sizeImage), interpolation=cv2.INTER_CUBIC) for x in X_train])
        X_test_resized = np.array([cv2.resize(x, (sizeImage, sizeImage), interpolation=cv2.INTER_CUBIC) for x in X_test])
    elif resize_meth == "bilinear":
        X_train_resized = np.array([cv2.resize(x, (sizeImage, sizeImage)) for x in X_train])
        X_test_resized = np.array([cv2.resize(x, (sizeImage, sizeImage)) for x in X_test])
    else:
        raise ValueError("Resizing method is not defined")

    # Repeat channel if grayscale
    if len(X_train[0].shape) == 2:
        X_train_resized = repeatChannel(X_train_resized, 3)
    if len(X_test[0].shape) == 2:
        X_test_resized = repeatChannel(X_test_resized, 3)

    # Reshape
    X_train = X_train_resized.reshape(-1, sizeImage, sizeImage, 3)
    X_test = X_test_resized.reshape(-1, sizeImage, sizeImage, 3)

    # Train-validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=sizeTest, random_state=8, stratify=y_train
    )

    return X_train, y_train, X_val, y_val, X_test, y_test, y_test_label


#python InferenceEnsembelModifiedCNNV1.py --dataset ICFHR18_OS --nb_class 60 --resize_meth bilinear --color_pad gray_white --img_size 75  --path_out expExpModifiedCNNV1Ensembel_gray_white --weight 'imagenet'
def main():
    parser = argparse.ArgumentParser(description="Process resize method and image size.")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g., ICFHR18_OS_S, ICFHR18_OS_MC, ICFHR18_OS)")
    parser.add_argument("--nb_class", type=int, required=True, help="number of class (e.g., 7, 57, 60, 111, 133)")
    parser.add_argument("--resize_meth", type=str, required=True, help="Resize method to use (e.g., bilinear, bicubic, nearest)")
    parser.add_argument("--color_pad", type=str, required=True, help="Color padding (e.g., white, black, transparent)")
    parser.add_argument("--img_size", type=int, required=True, help="Image size (e.g., 32, 64, 75, 80, 128)")
    parser.add_argument("--path_out", type=str, required=True, help="Output path")
    
    parser.add_argument("--weight", type=str, default='imagenet',  required=False, help="weight_name (imagenet, noTL)")

    
    args = parser.parse_args()

    
    
    model_names = [
        'EfficientNetV2S', 'EfficientNetV2M', 'EfficientNetV2L'
    ]

    # === Path ke model H5 ===
    model_paths = {
        'EfficientNetV2S': f'./models_20250407_V1/imagenet-{args.dataset}_gray_white_75-EfficientNetV2S-size75-batch32/last_model.h5',
        'EfficientNetV2M': f'./models_20250407_V1/imagenet-{args.dataset}_gray_white_75-EfficientNetV2M-size75-batch32/last_model.h5',
        'EfficientNetV2L': f'./models_20250407_V1/imagenet-{args.dataset}_gray_white_75-EfficientNetV2L-size75-batch32/last_model.h5',
    }

    print(f"Dataset: {args.dataset}")
    print(f"Resize Method: {args.resize_meth}")
    print(f"Color Padding: {args.color_pad}")
    print(f"Image Size: {args.img_size}")
    
    print(f"weight: {args.weight}")

    if '_OK_' in args.dataset:
        # Extract the part before '_OK_'
        match = re.match(r'(.+)_OK_', args.dataset)
        if match:
            return match.group(1) + '_OK'

    path_code = f'{args.dataset}_{args.color_pad}_{args.img_size}'
    
    nb_classes = args.nb_class

    # or None for no transfer learning
    if args.weight == "noTL":
        weight = None  
    elif args.weight == "imagenet":
        weight = args.weight  
    # print(weight)
    # Training parameters     
    path_result = f'{args.path_out}_{args.resize_meth}_{args.img_size}_{args.dataset}'
    sizeImage = args.img_size
    batch_size = 32
    epochs = 100
    # Model selection
    MODEL = ['VGG19', 'ResNet50V2', 'ResNet101V2', 'ResNet152V2', 'InceptionResNetV2', 'MobileNet', 'MobileNetV2',
            'NASNetMobile', 'EfficientNetV2B0', 'EfficientNetV2B3', 'EfficientNetV2S', 'EfficientNetV2M', 'EfficientNetV2L',
            'MobileEfV2S','IncResnetV2EfV2S','CrossAttentionFusion','AdaptiveCrossAttentionFusion']

    
    

    if not os.path.exists(path_result):
        os.makedirs(path_result)
    # Load data
    if args.dataset == "mnist":
        pathTrain = 'mnist'
        pathTest = 'mnist'
        X_train, y_train, X_val, y_val, X_test, y_test, y_test_label = utils.load_mnist_train_test_split_custom(sizeImage, 0.2, args.resize_meth)
    elif args.dataset == "omniglot":        
        pathTrain = './datasets/omniglot_9_1_train.npz'
        pathTest = './datasets/omniglot_9_1_test.npz'
        X_train, y_train, X_val, y_val, X_test, y_test, y_test_label = utils.load_npz_train_test_split(
            pathTrain, pathTest, sizeImage, 0.2
        )
    elif args.dataset == "OS_ICFHR":        
        pathTrain = './datasets/OS_ICFHR_train_gray.npz'
        pathTest = './datasets/OS_ICFHR_test_gray.npz'
        X_train, y_train, X_val, y_val, X_test, y_test, y_test_label = utils.load_npz_train_test_split(
            pathTrain, pathTest, sizeImage, 0.2
        )
    elif args.dataset == "ICFHR18_OS_M":        
        pathTrain = './datasets/OS_ICFHR18_57_train_gray_white.npz'
        pathTest = './datasets/OS_ICFHR18_57_test_gray_white.npz'
        X_train, y_train, X_val, y_val, X_test, y_test, y_test_label = utils.load_npz_train_test_split(
            pathTrain, pathTest, sizeImage, 0.2
        )
    else :    
        pathTrain = f'./datasets/{args.dataset}_train_set_{args.color_pad}_224.npz'
        pathTest = f'./datasets/{args.dataset}_test_set_{args.color_pad}_224.npz'
        X_train, y_train, X_val, y_val, X_test, y_test, y_test_label = load_npz_train_test_split_custom(
            pathTrain, pathTest, sizeImage, 0.2, args.resize_meth
        )
        


    predictions = []
    # Evaluate the model using the best model
    start_time = time.time()
    for model_name in model_names:
        print(f"🔍 Loading model: {model_name}")

        X_train, X_val, X_test = apply_preprocess_input(X_train, X_val, X_test, model_name)
        Y_test = to_categorical(y_test, nb_classes)
        input_shape = X_test.shape[1:]
        # model = models.build_MobileNet_keras(X_train.shape[1:], nb_classes, weight)
        if model_name == 'EfficientNetV2S' : 
            model = models.build_EfficientNetV2S_keras(input_shape, nb_classes, weight)
        elif model_name == 'EfficientNetV2M' : 
            model = models.build_EfficientNetV2M_keras(input_shape, nb_classes, weight)
        elif model_name == 'EfficientNetV2L' :        
            model = models.build_EfficientNetV2L_keras(input_shape, nb_classes,weight)
        
        model.load_weights(model_paths[model_name]) # ⬅️ Tambahkan ini
        # model = load_model(model_paths[model_name])

        preprocess_fn = model_preprocessors[model_name]
        X_preprocessed = preprocess_fn(X_test.copy())
        y_pred_proba = np.vstack([model.predict(X_preprocessed[i:i + batch_size]) for i in range(0, len(X_preprocessed), batch_size)])
        
        y_pred_class = np.argmax(y_pred_proba, axis=1)
        y_true_class = np.argmax(Y_test, axis=1)
        top1_acc = np.mean(y_pred_class == y_true_class)
        print("Top-1 Accuracy:", top1_acc)

        predictions.append(y_pred_proba)

    # Ensemble via softmax averaging    
    testing_time = time.time() - start_time
    # Ensemble via soft voting (average softmax)
    ensemble_pred = np.mean(predictions, axis=0)  # shape: (num_samples, num_classes)

    # Predicted class from ensemble
    y_pred_class = np.argmax(ensemble_pred, axis=1)
    y_true_class = np.argmax(Y_test, axis=1)

    # Top-1 accuracy
    top1_acc = np.mean(y_pred_class == y_true_class)

    # Top-5 accuracy
    top5_preds = np.argsort(ensemble_pred, axis=1)[:, -5:]  # ambil 5 kelas paling tinggi
    top5_acc = np.mean([true in pred for true, pred in zip(y_true_class, top5_preds)])

    # AUC Scores
    macro_auc = np.mean([roc_auc_score(Y_test[:, j], ensemble_pred[:, j]) for j in range(nb_classes)])
    micro_auc = roc_auc_score(Y_test, ensemble_pred, average='weighted', multi_class='ovr')

    # F1 Scores
    macro_f1 = f1_score(y_true_class, y_pred_class, average='macro')
    weighted_f1 = f1_score(y_true_class, y_pred_class, average='weighted')

    # Balanced accuracy and MCC
    balanced_acc = balanced_accuracy_score(y_true_class, y_pred_class)
    mcc = matthews_corrcoef(y_true_class, y_pred_class)

    # Log results
    log_csv_path = f'./{path_result}/results_training_best_model.csv'
    log_entry = f"{datetime.now()};{pathTrain};{args.weight};Ensembel;{testing_time:.4f};{sizeImage};{epochs};{batch_size};"
    log_entry += f"Last;{args.resize_meth};{args.color_pad};{top1_acc:.4f};{top5_acc:.4f};{macro_auc:.4f};{micro_auc:.4f};{macro_f1:.4f};{weighted_f1:.4f};{balanced_acc:.4f};{mcc:.4f}\n"

    if not os.path.exists(log_csv_path):
        with open(log_csv_path, 'w') as f:
            f.write("datetime;pathTrain;weight;model;testing_time;sizeImage;epochs;batch_size;typeModel;method;color_pad;top1_acc;top5_acc;macro_auc;micro_auc;macro_f1;weighted_f1;balanced_acc;mcc\n")


    with open(log_csv_path, 'a') as f:
        f.write(log_entry)

    # Merge all results
    log_csv_path = f'./{args.path_out}_{args.dataset}_{args.weight}_results_training.csv'
    log_entry = f"{datetime.now()};{pathTrain};{args.weight};Ensembel;{testing_time:.4f};{sizeImage};{epochs};{batch_size};"
    log_entry += f"Best;{args.resize_meth};{args.color_pad};{top1_acc:.4f};{top5_acc:.4f};{macro_auc:.4f};{micro_auc:.4f};{macro_f1:.4f};{weighted_f1:.4f};{balanced_acc:.4f};{mcc:.4f}\n"

    if not os.path.exists(log_csv_path):
        with open(log_csv_path, 'w') as f:
            f.write("datetime;pathTrain;weight;model;testing_time;sizeImage;epochs;batch_size;typeModel;method;color_pad;top1_acc;top5_acc;macro_auc;micro_auc;macro_f1;weighted_f1;balanced_acc;mcc\n")


    with open(log_csv_path, 'a') as f:
        f.write(log_entry)

    print("Top-1 Accuracy:", top1_acc)
    print("Balanced Accuracy:", balanced_acc)

   # === Load label mapping from test file ===
    # dataset_test_npz = np.load(pathTest)
    # df_test_map = pd.DataFrame({
    #     'label_code': dataset_test_npz['label_code'],
    #     'label': dataset_test_npz['label'].astype(str)
    # })
    df_test_map = pd.DataFrame({
        'label_code': y_test,
        'label': y_test_label.astype(str)
    })
    df_unik = df_test_map.drop_duplicates().sort_values(by='label_code')
    label_mapping = dict(zip(df_unik['label_code'], df_unik['label']))

    # === Convert y_true and y_pred from indices to label names ===
    y_true_labels = pd.Series(y_true_class).replace(label_mapping).tolist()
    y_pred_labels = pd.Series(y_pred_class).replace(label_mapping).tolist()

    # === Confusion Matrix: Count-Based ===
    df_confusion_count = pd.crosstab(
        pd.Series(y_true_labels, name='Actual'),
        pd.Series(y_pred_labels, name='Predicted')
    )

    cm_count_csv_path = f"./{args.path_out}_{args.dataset}_{args.weight}_confusion_matrix_count.csv"
    df_confusion_count.to_csv(cm_count_csv_path)
    print(f"✅ Confusion matrix (count) saved to: {cm_count_csv_path}")

    # === Confusion Matrix: Percent-Based (per row normalization) ===
    df_confusion_percent = pd.crosstab(
        pd.Series(y_true_labels, name='Actual'),
        pd.Series(y_pred_labels, name='Predicted'),
        normalize='index'
    ) * 100
    df_confusion_percent = df_confusion_percent.round(2)

    cm_percent_csv_path = f"./{args.path_out}_{args.dataset}_{args.weight}_confusion_matrix_percent.csv"
    df_confusion_percent.to_csv(cm_percent_csv_path)
    print(f"✅ Confusion matrix (percent) saved to: {cm_percent_csv_path}")

    # === Heatmap Helper Function ===
    def save_heatmap(df_cm, title, file_prefix, fmt_val):
        num_classes = len(df_cm)
        fig_width = max(12, int(num_classes * 0.4))
        fig_height = max(8, int(num_classes * 0.3))

        plt.figure(figsize=(fig_width, fig_height))
        sns.heatmap(
            df_cm,
            annot=True,
            fmt=fmt_val,
            cmap='Blues',
            cbar=True,
            annot_kws={"size": 6}
        )
        plt.title(title)
        plt.ylabel("Actual")
        plt.xlabel("Predicted")
        plt.xticks(rotation=90, fontsize=6)
        plt.yticks(rotation=0, fontsize=6)
        plt.tight_layout()

        png_path = f"./{file_prefix}.png"
        pdf_path = f"./{file_prefix}.pdf"
        plt.savefig(png_path, dpi=300)
        plt.savefig(pdf_path)
        plt.close()
        print(f"🖼️ Confusion matrix heatmap saved to: {png_path} and {pdf_path}")

    # === Save Heatmaps ===
    save_heatmap(
        df_confusion_count,
        title=f"{args.dataset} – Confusion Matrix (Count per Actual Class)",
        file_prefix=f"{args.path_out}_{args.dataset}_{args.weight}_confusion_matrix_count",
        fmt_val='d'
    )

    save_heatmap(
        df_confusion_percent,
        title=f"{args.dataset} – Confusion Matrix (Percent per Actual Class)",
        file_prefix=f"{args.path_out}_{args.dataset}_{args.weight}_confusion_matrix_percent",
        fmt_val='.1f'
    )

    # === Save images with wrong predictions ===
    output_dir = f"./{args.path_out}_{args.dataset}_misclassified"
    
    os.makedirs(output_dir, exist_ok=True)

    for i, (true_label, pred_label) in enumerate(zip(y_true_labels, y_pred_labels)):
        if true_label != pred_label:
            img = X_test[i]
            
            # If image is flat, reshape (assuming square and grayscale)
            if img.ndim == 1:
                side_len = int(np.sqrt(img.shape[0]))
                img = img.reshape((side_len, side_len))
            
            plt.imshow(img, cmap='gray')  # adjust cmap for RGB if needed
            plt.axis('off')
            
            # Safe filename (avoid slashes, etc.)
            filename = f"{i}_{true_label.replace('/', '_')} - {pred_label.replace('/', '_')}.png"
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, bbox_inches='tight', pad_inches=0)
            plt.close()
    



if __name__ == "__main__":
    main()

