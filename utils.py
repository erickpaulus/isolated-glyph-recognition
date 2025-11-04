import matplotlib.pyplot as plt
import numpy as np
import cv2
import seaborn as sn
from sklearn.model_selection import train_test_split
import tensorflow as tf


def preprocess_dataset(nb_classes,is_training=True):
    def _pp(image, label):
        # if is_training:
        #     # Resize to a bigger spatial resolution and take the random
        #     # crops.
        #     image = tf.image.resize(image, (resize_bigger, resize_bigger))
        #     # image = tf.image.random_crop(image, (image_size, image_size, 3))
        #     image = tf.image.random_flip_left_right(image)
        # else:
        #     image = tf.image.resize(image, (image_size, image_size))
        label = tf.one_hot(label, depth=nb_classes)
        return image, label

    return _pp


def prepare_dataset(dataset, batch_size, nb_classes, AUTOTUNE, is_training=True):
    
    if is_training:
        dataset = dataset.shuffle(batch_size * 10)
    dataset = dataset.map(preprocess_dataset(nb_classes, is_training), num_parallel_calls=AUTOTUNE)
    # return dataset.batch(batch_size).prefetch(auto)
    return dataset

def load_images(data_dir,sizeImage,batch_size,nb_classes):
    

    train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(sizeImage, sizeImage),
    batch_size=batch_size)

    val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(sizeImage, sizeImage),
    batch_size=batch_size)

    class_names = train_ds.class_names
    print(class_names)

    # normalization_layer = tf.keras.layers.Rescaling(1./255)
    # normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    # image_batch, labels_batch = next(iter(normalized_ds))
    # first_image = image_batch[0]
    # Notice the pixel values are now in `[0,1]`.
    # print(np.min(first_image), np.max(first_image))

    AUTOTUNE = tf.data.AUTOTUNE

    train_dataset = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_dataset = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    train_dataset = prepare_dataset(train_dataset, batch_size, nb_classes, AUTOTUNE,  is_training=True)
    val_dataset = prepare_dataset(val_dataset, batch_size, nb_classes, AUTOTUNE,  is_training=False)
    
    return train_dataset, val_dataset

def load_npz_split(pathTrain, pathTest,sizeImage,sizeTest):
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

    # Resize the images to 224x224 and expand the dimensionality to (sizeImage, sizeImage, channel)
    X_train_resized = np.array([cv2.resize(x, dsize=(sizeImage, sizeImage)) for x in X_train])
    X_test_resized = np.array([cv2.resize(x, dsize=(sizeImage, sizeImage)) for x in X_test])
    
    # add an additional dimension to represent the single-channel
    X_train = X_train_resized.reshape(-1, sizeImage, sizeImage, 1)
    X_test = X_test_resized.reshape(-1, sizeImage, sizeImage, 1)
    # X = normalize_images(X)
    
    # one-hot format classes    
    # y_train = np_utils.to_categorical(y_train)
    # y_test = np_utils.to_categorical(y_test)

    # normalize each value for each pixel for the entire vector for each input
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    # Use the same function above for the validation set 20%
    # Setting the random_state is desirable for reproducibility.
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=sizeTest, random_state= 8,stratify = y_train) 

    return X_train, y_train, X_val, y_val, X_test, y_test, y_test_label

def load_npz_train_test_split(pathTrain, pathTest,sizeImage,sizeTest):
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

    print("Len X_train shape:", len(X_train[0].shape))

    dataset = np.load(pathTest)
    X_test = dataset['data']
    y_test = dataset['label_code']
    y_test_label = dataset['label']
    print("Len X_train shape:", len(X_test[0].shape))
    

    # Resize the images to 224x224 and expand the dimensionality to (sizeImage, sizeImage, channel)
    X_train_resized = np.array([cv2.resize(x, dsize=(sizeImage, sizeImage), interpolation=cv2.INTER_NEAREST) for x in X_train])
    X_test_resized = np.array([cv2.resize(x, dsize=(sizeImage, sizeImage), interpolation=cv2.INTER_NEAREST) for x in X_test])
    #X_train_resized = np.array([cv2.resize(x, dsize=(sizeImage, sizeImage), interpolation=cv2.INTER_CUBIC) for x in X_train])
    #X_test_resized = np.array([cv2.resize(x, dsize=(sizeImage, sizeImage), interpolation=cv2.INTER_CUBIC) for x in X_test])
    
    #X_train_resized = np.array([cv2.resize(x, dsize=(sizeImage, sizeImage)) for x in X_train])
    #X_test_resized = np.array([cv2.resize(x, dsize=(sizeImage, sizeImage)) for x in X_test])
    # repeat channel
    if len(X_train[0].shape) == 2:
        X_train_resized = repeatChannel(X_train_resized,3)    
    if len(X_test[0].shape) == 2:
        X_test_resized = repeatChannel(X_test_resized,3)

    print(" X_train_resized shape:", X_train_resized[0].shape)
    # add an additional dimension to represent the single-channel
    X_train = X_train_resized.reshape(-1, sizeImage, sizeImage, 3)
    X_test = X_test_resized.reshape(-1, sizeImage, sizeImage, 3)
    # X = normalize_images(X)
    
    # one-hot format classes    
    # y_train = np_utils.to_categorical(y_train)
    # y_test = np_utils.to_categorical(y_test)

    # normalize each value for each pixel for the entire vector for each input
    # X_train = X_train / 255.0
    # X_test = X_test / 255.0

    # Use the same function above for the validation set 20%
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=sizeTest, random_state= 8,stratify = y_train) 

    return X_train, y_train, X_val, y_val, X_test, y_test, y_test_label

def load_npz_train_test_split_channel(pathTrain, pathTest,sizeImage,sizeTest,channel):
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

    # Convert RGB to grayscale by taking the mean across the last dimension
    # X_train = np.mean(X_train, axis=-1)
    # X_test = np.mean(X_test, axis=-1)
    


    # # Resize the images to 224x224 and expand the dimensionality to (sizeImage, sizeImage, channel)
    # X_train_resized = np.array([cv2.resize(x, dsize=(sizeImage, sizeImage), interpolation=cv2.INTER_NEAREST) for x in X_train])
    # X_test_resized = np.array([cv2.resize(x, dsize=(sizeImage, sizeImage), interpolation=cv2.INTER_NEAREST) for x in X_test])
    # #X_train_resized = np.array([cv2.resize(x, dsize=(sizeImage, sizeImage), interpolation=cv2.INTER_CUBIC) for x in X_train])
    # #X_test_resized = np.array([cv2.resize(x, dsize=(sizeImage, sizeImage), interpolation=cv2.INTER_CUBIC) for x in X_test])
    
    # #X_train_resized = np.array([cv2.resize(x, dsize=(sizeImage, sizeImage)) for x in X_train])
    # #X_test_resized = np.array([cv2.resize(x, dsize=(sizeImage, sizeImage)) for x in X_test])
    # # repeat channel
    # if len(X_train[0].shape) == 2:
    #     X_train_resized = repeatChannel(X_train_resized,channel)    
    # if len(X_test[0].shape) == 2:
    #     X_test_resized = repeatChannel(X_test_resized,channel)

    # # add an additional dimension to represent the single-channel
    # X_train = X_train_resized.reshape(-1, sizeImage, sizeImage, channel)
    # X_test = X_test_resized.reshape(-1, sizeImage, sizeImage, channel)
    # # X = normalize_images(X)
    
    # # one-hot format classes    
    # # y_train = np_utils.to_categorical(y_train)
    # # y_test = np_utils.to_categorical(y_test)

    # # normalize each value for each pixel for the entire vector for each input
    # X_train = X_train / 255.0
    # X_test = X_test / 255.0

    # Use the same function above for the validation set 20%
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=sizeTest, random_state= 8,stratify = y_train) 

    return X_train, y_train, X_val, y_val, X_test, y_test, y_test_label

import keras
def load_mnist_train_test_split(sizeImage,sizeTest):
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
    # Load the data and split it between train and test sets
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
    
    y_test_label = y_test

    

    # Resize the images to 224x224 and expand the dimensionality to (sizeImage, sizeImage, channel)
    X_train_resized = np.array([cv2.resize(x, dsize=(sizeImage, sizeImage), interpolation=cv2.INTER_NEAREST) for x in X_train])
    X_test_resized = np.array([cv2.resize(x, dsize=(sizeImage, sizeImage), interpolation=cv2.INTER_NEAREST) for x in X_test])
    
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

    # Use the same function above for the validation set 20%
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=sizeTest, random_state= 8,stratify = y_train) 

    return X_train, y_train, X_val, y_val, X_test, y_test, y_test_label

def load_mnist_train_test_split_custom(sizeImage,sizeTest, resize_meth):
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
    # Load the data and split it between train and test sets
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
    
    y_test_label = y_test

    

    # Resize the images to 224x224 and expand the dimensionality to (sizeImage, sizeImage, channel)
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

    # Use the same function above for the validation set 20%
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=sizeTest, random_state= 8,stratify = y_train) 

    return X_train, y_train, X_val, y_val, X_test, y_test, y_test_label


# print(history.history.keys())
# summarize history for accuracy
def visualize_result(history, fn_out):  
  plt.figure(1)
  plt.plot(history.history['accuracy'])
  plt.plot(history.history['val_accuracy'])
  plt.title('model accuracy')
  plt.ylabel('accuracy')
  plt.xlabel('epoch')
  plt.legend(['train', 'test'], loc='upper left')
  plt.savefig(fn_out + "_model accuracy.png")
  # plt.show()
  # summarize history for loss
  plt.figure(2)
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.title('model loss')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.legend(['train', 'test'], loc='upper left')
  plt.savefig(fn_out + "_model loss.png")
  # plt.show()



def plot_confusion_matrix_values(df_confusion,fn_out, title='Confusion matrix', cmap=plt.cm.gray_r):
  plt.figure(figsize = (15,10))
  plt.title(title)
  sn.heatmap(df_confusion, annot=True,cmap=plt.cm.gray_r)
  plt.savefig(fn_out + "_Confusion_matrix.png")
  # plt.show()

# def plot_confusion_matrix(df_confusion, title='Confusion matrix', cmap=plt.cm.gray_r):
#     plt.figure("Figure Confusion matrix")
#     plt.matshow(df_confusion, cmap=cmap) # imshow
#     plt.title(title)
#     plt.colorbar()
#     tick_marks = np.arange(len(df_confusion.columns))
#     plt.xticks(tick_marks, df_confusion.columns, rotation=45)
#     plt.yticks(tick_marks, df_confusion.index)
#     plt.tight_layout()
#     plt.ylabel(df_confusion.index.name)
#     plt.xlabel(df_confusion.columns.name)
#     plt.figure(figsize=(20, 20))

def repeatChannel(x, channel):
    # Reshape the images to 3D arrays with a grayscale channel
    x = np.expand_dims(x, axis=-1)    

    # Repeat the grayscale channel three times to create 3-channel images
    x = np.repeat(x, channel, axis=-1)
    return x

# Define a function to change the MNIST label dimension
def change_to_right(wrong_labels):
    right_labels=[]
    for x in wrong_labels:
        for i in range(0,len(wrong_labels[0])):
            if x[i]==1:
                right_labels.append(i)
    return right_labels

def select_data(x_train, y_train, num_small_class, num_large_class):
    selected_indices_class_1 = np.where(y_train == 1)[0][:num_small_class]
    selected_indices_class_2 = np.where(y_train == 2)[0][:num_small_class]

    # Select the remaining samples for other classes
    remaining_indices = np.setdiff1d(range(len(y_train)), np.concatenate([selected_indices_class_1, selected_indices_class_2]))

    selected_indices_rest = np.random.choice(remaining_indices, num_large_class, replace=False)

    # Combine selected indices
    selected_indices = np.concatenate([selected_indices_class_1, selected_indices_class_2, selected_indices_rest])
    return selected_indices