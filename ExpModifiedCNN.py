import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf
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
K.clear_session()
try:
    subprocess.check_output('nvidia-smi')
    print('Nvidia GPU detected!')
    print(tf.config.list_physical_devices('GPU'))
except Exception: # this command not being found can raise quite a few different errors depending on the configuration
    print('No Nvidia GPU in system!')

import argparse

def repeatChannel(x, channel):
    # Reshape the images to 3D arrays with a grayscale channel
    x = np.expand_dims(x, axis=-1)    

    # Repeat the grayscale channel three times to create 3-channel images
    x = np.repeat(x, channel, axis=-1)
    return x

def load_npz_train_test_split_custom(pathTrain, pathTest,sizeImage,sizeTest, resize_meth):
    '''
    Load  data sets for training, validation, and test.
    default size of image is 28 x 28 x D
    Args:
        path

    Returns:
        (x_train, y_train): (4-D array, 2-D array)
        (x_val, y_val): (4-D array, 2-D array)
        (x_test, y_test): (4-D array, 2-D array)
    '''
    dataset = np.load(pathTrain)
    X_train = dataset['data']
    y_train = dataset['label_code']

    dataset = np.load(pathTest)
    X_test = dataset['data']
    y_test = dataset['label_code']
    y_test_label = dataset['label']

    if resize_meth == "nearest":    
        X_train_resized = np.array([cv2.resize(x, dsize=(sizeImage, sizeImage), interpolation=cv2.INTER_NEAREST) for x in X_train])
        X_test_resized = np.array([cv2.resize(x, dsize=(sizeImage, sizeImage), interpolation=cv2.INTER_NEAREST) for x in X_test])
    elif resize_meth == "bicubic":    
        X_train_resized = np.array([cv2.resize(x, dsize=(sizeImage, sizeImage), interpolation=cv2.INTER_CUBIC) for x in X_train])
        X_test_resized = np.array([cv2.resize(x, dsize=(sizeImage, sizeImage), interpolation=cv2.INTER_CUBIC) for x in X_test])
    elif resize_meth == "bilinear": 
        X_train_resized = np.array([cv2.resize(x, dsize=(sizeImage, sizeImage)) for x in X_train])
        X_test_resized = np.array([cv2.resize(x, dsize=(sizeImage, sizeImage)) for x in X_test])
    else:
        raise ValueError("Resizing method is not defined")
    
    # repeat channel
    if len(X_train[0].shape) == 2:
        X_train_resized = repeatChannel(X_train_resized,3)    
    if len(X_test[0].shape) == 2:
        X_test_resized = repeatChannel(X_test_resized,3)

    # add an additional dimension to represent the single-channel
    X_train = X_train_resized.reshape(-1, sizeImage, sizeImage, 3)
    X_test = X_test_resized.reshape(-1, sizeImage, sizeImage, 3)
    # X = normalize_images(X)
    
    # one-hot format classes    
    # y_train = np_utils.to_categorical(y_train)
    # y_test = np_utils.to_categorical(y_test)

    # normalize each value for each pixel for the entire vector for each input
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    max_value = np.max(X_train)
    print("Max value in X_train:", max_value)
    
    # Use the same function above for the validation set 20%
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=sizeTest, random_state= 8,stratify = y_train) 

    return X_train, y_train, X_val, y_val, X_test, y_test, y_test_label

def main():
    parser = argparse.ArgumentParser(description="Process resize method and image size.")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g., ICFHR18_OS_S, ICFHR18_OS_MC, ICFHR18_OS)")
    parser.add_argument("--nb_class", type=int, required=True, help="number of class (e.g., 7, 57, 60, 111, 133)")
    parser.add_argument("--resize_meth", type=str, required=True, help="Resize method to use (e.g., bilinear, bicubic, nearest)")
    parser.add_argument("--color_pad", type=str, required=True, help="Color padding (e.g., white, black, transparent)")
    parser.add_argument("--img_size", type=int, required=True, help="Image size (e.g., 32, 64, 75, 80, 128)")
    parser.add_argument("--path_out", type=str, required=True, help="Output path")
    parser.add_argument("--model_index", type=int, default=5,  required=False,  help= (
            "Model index: " 
            "0=VGG19, 1=ResNet50V2, 2=ResNet101V2, 3=ResNet152V2, "
            "4=InceptionResNetV2, 5=MobileNet, 6=MobileNetV2, 7=NASNetMobile, "
            "8=EfficientNetV2B0, 9=EfficientNetV2B3, 10=EfficientNetV2S, "
            "11=EfficientNetV2M, 12=EfficientNetV2L."
        ),
    )
    parser.add_argument("--weight", type=str, default='imagenet',  required=False, help="weight_name (imagenet, noTL)")

    
    args = parser.parse_args()
    print(f"Dataset: {args.dataset}")
    print(f"Resize Method: {args.resize_meth}")
    print(f"Color Padding: {args.color_pad}")
    print(f"Image Size: {args.img_size}")
    print(f"model_index: {args.model_index}")
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
    print(weight)
    # Training parameters     
    path_result = f'{args.path_out}_{args.resize_meth}_{args.img_size}_{args.dataset}'
    sizeImage = args.img_size
    batch_size = 32
    epochs = 100
    

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

    print("X_train shape", X_train.shape)
    print("y_train shape", y_train.shape)
    print("X_val shape", X_val.shape)
    print("y_val shape", y_val.shape)
    print("X_test shape", X_test.shape)
    print("y_test shape", y_test.shape)
    print("y_test_label shape", y_test_label.shape)

    # One-hot encoding labels
    Y_train = to_categorical(y_train, nb_classes)
    Y_val = to_categorical(y_val, nb_classes)
    Y_test = to_categorical(y_test, nb_classes)

    # Data augmentation
    gen = ImageDataGenerator(rotation_range=8, width_shift_range=0.08, shear_range=0.3,
                            height_shift_range=0.08, zoom_range=0.08)
    test_gen = ImageDataGenerator()

    train_generator = gen.flow(X_train, Y_train, batch_size=batch_size)
    val_generator = test_gen.flow(X_val, Y_val, batch_size=batch_size)
    test_generator = test_gen.flow(X_test, Y_test, batch_size=batch_size)

    # Model selection
    MODEL = ['VGG19', 'ResNet50V2', 'ResNet101V2', 'ResNet152V2', 'InceptionResNetV2', 'MobileNet', 'MobileNetV2',
            'NASNetMobile', 'EfficientNetV2B0', 'EfficientNetV2B3', 'EfficientNetV2S', 'EfficientNetV2M', 'EfficientNetV2L',
            'MobileEfV2S','IncResnetV2EfV2S','CrossAttentionFusion','AdaptiveCrossAttentionFusion']

    model_index = args.model_index  # Choosing MobileNet
    input_shape = X_train.shape[1:]
    # model = models.build_MobileNet_keras(X_train.shape[1:], nb_classes, weight)
    if model_index == 0 : 
        model = models.build_VGG19_keras(input_shape,nb_classes,weight)
    elif model_index == 1:
        model = models.build_ResNet50V2_keras(input_shape, nb_classes,weight)
    elif model_index == 2 :        
        model =models.build_ResNet101V2_keras(input_shape, nb_classes,weight)
    elif model_index == 3 :
        model = models.build_ResNet152V2_keras(input_shape, nb_classes,weight)
    elif model_index == 4 :
        model = models.build_InceptionResNetV2_keras(input_shape, nb_classes,weight)
    elif model_index == 5 :
        model = models.build_MobileNet_keras(input_shape, nb_classes, weight)
    elif model_index == 6 :
        model = models.build_MobileNetV2_keras(input_shape, nb_classes, weight)
    elif model_index == 7 :
        model = models.build_NASNetMobile_keras(input_shape, nb_classes, weight)
    elif model_index == 8 :
        model = models.build_EfficientNetV2B0_keras(input_shape, nb_classes, weight)
    elif model_index == 9 :
        model = models.build_EfficientNetV2B3_keras(input_shape, nb_classes, weight)
    elif model_index == 10 :
        model = models.build_EfficientNetV2S_keras(input_shape, nb_classes, weight)
    elif model_index == 11 :
        model = models.build_EfficientNetV2M_keras(input_shape, nb_classes, weight)
    elif model_index == 12 :
        model = models.build_EfficientNetV2L_keras(input_shape, nb_classes, weight)
    elif model_index == 13 :
        model = models.build_MobiEff_keras(input_shape, nb_classes, weight)
    elif model_index == 14 :
        model = models.build_IncResnetV2EfV2S_keras(input_shape, nb_classes, weight)
    elif model_index == 15 :
        model = models.build_CrossAttentionFusion_keras(input_shape, nb_classes, weight)
    elif model_index == 16 :
        model = models.build_AdaptiveCrossAttentionFusion_keras(input_shape, nb_classes, weight)
        
        
        

    # Compile model
    # lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    # initial_learning_rate=0.001, decay_steps=1000, decay_rate=0.9)
    # optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9)
    # model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    

    
    # Define output directory
    output_dir = f'./{path_result}/{args.weight}-{path_code}-{MODEL[model_index]}-size{sizeImage}-batch{batch_size}/'
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_filepath = os.path.join(output_dir, "best_model.weights.h5")

    # Callbacks
    callbacks = [
        ModelCheckpoint(checkpoint_filepath, save_best_only=True, save_weights_only=True, monitor='val_accuracy', mode='max'),  
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, min_delta=1e-4, mode='auto'),
        # ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, min_delta=1e-4, mode='auto'),
        TensorBoard(log_dir=os.path.join(output_dir, 'logs'), histogram_freq=1)
    ]

    # Train the model
    start_time = time.time()
    history = model.fit(
        train_generator,
        steps_per_epoch=len(X_train) // batch_size,
        epochs=epochs,
        validation_data=val_generator,
        validation_steps=len(X_val) // batch_size,
        callbacks=callbacks,
        verbose=1
    )
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")

    # Save training history
    np.save(os.path.join(output_dir, "history.npy"), history.history)
    model.save(os.path.join(output_dir, "last_model.h5"))

    # Evaluate the model using the last model
    start_time = time.time()
    predictions = np.vstack([model.predict(X_test[i:i + batch_size]) for i in range(0, len(X_test), batch_size)])
    testing_time = time.time() - start_time

    # Compute evaluation metrics
    y_pred_class = np.argmax(predictions, axis=1)
    y_true_class = np.argmax(Y_test, axis=1)

    top1_acc = np.mean(y_pred_class == y_true_class)
    top5_acc = np.mean([1 if true in pred else 0 for true, pred in zip(y_true_class, np.argsort(predictions, axis=1)[:, -5:])])
    macro_auc = np.mean([roc_auc_score(Y_test[:, j], predictions[:, j]) for j in range(nb_classes)])
    micro_auc = roc_auc_score(Y_test, predictions, average='weighted', multi_class='ovr')
    macro_f1 = f1_score(y_true_class, y_pred_class, average='macro')
    weighted_f1 = f1_score(y_true_class, y_pred_class, average='weighted')
    balanced_acc = balanced_accuracy_score(y_true_class, y_pred_class)
    mcc = matthews_corrcoef(y_true_class, y_pred_class)

    # Log results
    log_csv_path = f'./{path_result}/results_training.csv'
    log_entry = f"{datetime.now()};{pathTrain};{args.weight};{MODEL[model_index]};{training_time:.2f};{testing_time:.4f};{sizeImage};{epochs};{batch_size};"
    log_entry += f"Last;{args.resize_meth};{args.color_pad};{top1_acc:.4f};{top5_acc:.4f};{macro_auc:.4f};{micro_auc:.4f};{macro_f1:.4f};{weighted_f1:.4f};{balanced_acc:.4f};{mcc:.4f}\n"

    if not os.path.exists(log_csv_path):
        with open(log_csv_path, 'w') as f:
            f.write("datetime;pathTrain;weight;model;training_time;testing_time;sizeImage;epochs;batch_size;typeModel;method;color_pad;top1_acc;top5_acc;macro_auc;micro_auc;macro_f1;weighted_f1;balanced_acc;mcc\n")

    with open(log_csv_path, 'a') as f:
        f.write(log_entry)

    # Merge all results
    log_csv_path = f'./{args.path_out}_{args.dataset}_{args.weight}_results_training.csv'
    log_entry = f"{datetime.now()};{pathTrain};{args.weight};{MODEL[model_index]};{training_time:.2f};{testing_time:.4f};{sizeImage};{epochs};{batch_size};"
    log_entry += f"Last;{args.resize_meth};{args.color_pad};{top1_acc:.4f};{top5_acc:.4f};{macro_auc:.4f};{micro_auc:.4f};{macro_f1:.4f};{weighted_f1:.4f};{balanced_acc:.4f};{mcc:.4f}\n"

    if not os.path.exists(log_csv_path):
        with open(log_csv_path, 'w') as f:
            f.write("datetime;pathTrain;weight;model;training_time;testing_time;sizeImage;epochs;batch_size;typeModel;method;color_pad;top1_acc;top5_acc;macro_auc;micro_auc;macro_f1;weighted_f1;balanced_acc;mcc\n")

    with open(log_csv_path, 'a') as f:
        f.write(log_entry)

    print("Top-1 Accuracy:", top1_acc)
    print("Balanced Accuracy:", balanced_acc)
    print(f"✅ Model {MODEL[model_index]} training and evaluation complete using the last model!")



    # Evaluate the model using the best model
    start_time = time.time()
    model.load_weights(checkpoint_filepath)  # ⬅️ Tambahkan ini
    predictions = np.vstack([model.predict(X_test[i:i + batch_size]) for i in range(0, len(X_test), batch_size)])
    testing_time = time.time() - start_time

    # Compute evaluation metrics
    y_pred_class = np.argmax(predictions, axis=1)
    y_true_class = np.argmax(Y_test, axis=1)

    top1_acc = np.mean(y_pred_class == y_true_class)
    top5_acc = np.mean([1 if true in pred else 0 for true, pred in zip(y_true_class, np.argsort(predictions, axis=1)[:, -5:])])
    macro_auc = np.mean([roc_auc_score(Y_test[:, j], predictions[:, j]) for j in range(nb_classes)])
    micro_auc = roc_auc_score(Y_test, predictions, average='weighted', multi_class='ovr')
    macro_f1 = f1_score(y_true_class, y_pred_class, average='macro')
    weighted_f1 = f1_score(y_true_class, y_pred_class, average='weighted')
    balanced_acc = balanced_accuracy_score(y_true_class, y_pred_class)
    mcc = matthews_corrcoef(y_true_class, y_pred_class)

    # Log results
    log_csv_path = f'./{path_result}/results_training_best_model.csv'
    log_entry = f"{datetime.now()};{pathTrain};{args.weight};{MODEL[model_index]};{training_time:.2f};{testing_time:.4f};{sizeImage};{epochs};{batch_size};"
    log_entry += f"Last;{args.resize_meth};{args.color_pad};{top1_acc:.4f};{top5_acc:.4f};{macro_auc:.4f};{micro_auc:.4f};{macro_f1:.4f};{weighted_f1:.4f};{balanced_acc:.4f};{mcc:.4f}\n"

    if not os.path.exists(log_csv_path):
        with open(log_csv_path, 'w') as f:
            f.write("datetime;pathTrain;weight;model;training_time;testing_time;sizeImage;epochs;batch_size;typeModel;method;color_pad;top1_acc;top5_acc;macro_auc;micro_auc;macro_f1;weighted_f1;balanced_acc;mcc\n")


    with open(log_csv_path, 'a') as f:
        f.write(log_entry)
    
    # Merge all results
    log_csv_path = f'./{args.path_out}_{args.dataset}_{args.weight}_results_training.csv'
    log_entry = f"{datetime.now()};{pathTrain};{args.weight};{MODEL[model_index]};{training_time:.2f};{testing_time:.4f};{sizeImage};{epochs};{batch_size};"
    log_entry += f"Best;{args.resize_meth};{args.color_pad};{top1_acc:.4f};{top5_acc:.4f};{macro_auc:.4f};{micro_auc:.4f};{macro_f1:.4f};{weighted_f1:.4f};{balanced_acc:.4f};{mcc:.4f}\n"

    if not os.path.exists(log_csv_path):
        with open(log_csv_path, 'w') as f:
            f.write("datetime;pathTrain;weight;model;training_time;testing_time;sizeImage;epochs;batch_size;typeModel;method;color_pad;top1_acc;top5_acc;macro_auc;micro_auc;macro_f1;weighted_f1;balanced_acc;mcc\n")


    with open(log_csv_path, 'a') as f:
        f.write(log_entry)

    print("Top-1 Accuracy:", top1_acc)
    print("Balanced Accuracy:", balanced_acc)
    print(f"✅ Model {MODEL[model_index]} training and evaluation complete using the best model!")

if __name__ == "__main__":
    main()

