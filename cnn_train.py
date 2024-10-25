import os
import json
from re import X
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
import matplotlib.pyplot as plt
from tensorflow.python.keras.backend import sparse_categorical_crossentropy
import tqdm
import wandb
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint
from tqdm.keras import TqdmCallback
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tensorflow.python.keras.utils.np_utils import to_categorical
import inflect
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

p = inflect.engine()    
name_of_dataset = "art"
wandb.init(project="cnn_sign", name=name_of_dataset)

DATASET_PATH = f"./mfcc_npy_{name_of_dataset}"
import tensorflow as tf
tf.random.set_seed(42)

def load_data(dataset_path):
    
    labels_mapping = {"four": 0, "three": 1, "five": 2, "seven": 3}
    data_sets = {
        "train": {"mfcc": [], "labels": []},
        "val": {"mfcc": [], "labels": []},
        "test": {"mfcc": [], "labels": []}
    }
    # Assume directory structure is ../mfcc_npy/train/0/, ../mfcc_npy/train/1/, etc.
    for split in ["train","val","test"]:
        split_path = os.path.join(dataset_path, split)
        if not os.path.isdir(split_path):
            continue
        
        for label_dir in os.listdir(split_path):
            label_path = os.path.join(split_path, label_dir)
            if not os.path.isdir(label_path):
                continue
            
            word_label = p.number_to_words(label_dir)  # Convert folder name from number to word
            mapped_label = labels_mapping.get(word_label, -1)  # Map word label to a number label
            
            for file in os.listdir(label_path):
                if file.endswith(".npy"):
                    file_path = os.path.join(label_path, file)
                    mfcc = np.load(file_path).T.tolist()
                    
                    data_sets[split]["mfcc"].append(mfcc)
                    data_sets[split]["labels"].append(mapped_label)
                    
    # with open(dataset_path, "r") as fp:
    #     data = json.load(fp)

    # convert list into numpy array
    
    X_train = np.array(data_sets["train"]["mfcc"])
    y_train = np.array(data_sets["train"]["labels"])
    X_valid = np.array(data_sets["val"]["mfcc"])
    y_valid = np.array(data_sets["val"]["labels"])
    X_test = np.array(data_sets["test"]["mfcc"])
    y_test = np.array(data_sets["test"]["labels"])
    return X_train, y_train, X_valid, y_valid, X_test, y_test


def prepare_dataset():

    # load the data
    print(DATASET_PATH)
    X_train, y_train, X_valid, y_valid, X_test, y_test= load_data(DATASET_PATH)

    # create the train/test split
    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, y, test_size=test_size)

    # # create the train/validation split
    # X_train, X_validation, y_train, y_validation = train_test_split(
    #     X_train, y_train, test_size=valisation_size)

    X_train = X_train[..., np.newaxis]
    X_test = X_test[..., np.newaxis]
    X_valid = X_valid[..., np.newaxis]

    return X_train, X_valid, X_test, y_train, y_valid, y_test


def plot_history(history):
    fig, axs = plt.subplots(2)

    # accuracy subplot
    axs[0].plot(history.history["accuracy"], label="CNN Train Accuracy")
    axs[0].plot(history.history["val_accuracy"], label="CNN Val Accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy Evaluation")

    axs[1].plot(history.history["loss"], label="CNN Train Error")
    axs[1].plot(history.history["val_loss"], label="CNN Val Error")
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Error Evaluation")

    plt.savefig('Evaluation_art2.png')

def build_model(input_shape):

    # create the model
    model = keras.Sequential()

    # first conv layer
    model.add(keras.layers.Conv2D(
        32, (3, 3), activation="relu", input_shape=input_shape))
    model.add(keras.layers.MaxPool2D(
        pool_size=(3, 3), strides=(2, 2), padding="same"))
    model.add(keras.layers.BatchNormalization())

    # second conv layer
    model.add(keras.layers.Conv2D(
        32, (3, 3), activation="relu", input_shape=input_shape))
    model.add(keras.layers.MaxPool2D(
        pool_size=(3, 3), strides=(2, 2), padding="same"))
    model.add(keras.layers.BatchNormalization())

    # third conv layer
    model.add(keras.layers.Conv2D(
        32, (2, 2), activation="relu", input_shape=input_shape))
    model.add(keras.layers.MaxPool2D(
        pool_size=(2, 2), strides=(2, 2), padding="same"))
    model.add(keras.layers.BatchNormalization())

    # flatten output and feed into dense layer
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation="relu"))
    model.add(keras.layers.Dropout(0.3))

    # output layer
    model.add(keras.layers.Dense(10, activation="softmax"))

    return model


def predict(model, X, y):

    X = X[np.newaxis, ...]

    prediction = model.predict(X)
    print("Prediction shape is: {}".format(prediction.shape))

    predicted_index = np.argmax(prediction, axis=1)
    print("Expected Index is {}, Predicted Index is {}".format(y, predicted_index))


if __name__ == "__main__":
    # Create train, validation and test set
    X_train, X_validation, X_test, y_train, y_validation, y_test = prepare_dataset()
    # print(X_train.shape)
    # Build the CNN network
    print("X_train size:", X_train.shape)
    print("X_validation size:", X_validation.shape)
    print("X_test size:", X_test.shape)
    
    input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])
    model = build_model(input_shape)

    # Compile the network
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer,
                  loss=sparse_categorical_crossentropy, metrics=["accuracy"])
    model.summary()
    checkpoint_path = os.path.join("models", "model.keras")

    # # Callback do ModelCheckpoint
    # model_checkpoint_callback = WandbModelCheckpoint(
    #     filepath=checkpoint_path,
    #     save_weights_only=True,
    #     monitor='val_accuracy',  # Ou 'accuracy' para acurácia de treinamento
    #     mode='max',
    #     save_best_only=True,
    #     verbose=1) 
    # Train  the CNN
    history = model.fit(
        X_train, y_train,
        validation_data=(X_validation, y_validation),
        epochs=50,
        batch_size=32,
        verbose=0,  # Desabilita a saída padrão do Keras e deixa o controle para o tqdm
        callbacks=[WandbMetricsLogger(), WandbModelCheckpoint(filepath=checkpoint_path), TqdmCallback(verbose=1)]
    )
    
    # Evaluate the CNN on the test set
    test_error, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
    print("Accuracy on test set is {}".format(test_accuracy))

    # Make prediction on a sample
    X = X_test[50]
    y = y_test[50]
    predict(model, X, y)
    # Make predictions on the test set
    predictions = model.predict(X_test)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = y_test
    class_labels = range(4)  # Assuming you have 10 classes. Adjust this as per your dataset.

    # Generate the confusion matrix
    cm = confusion_matrix(true_classes, predicted_classes, labels=class_labels)
    accuracy = accuracy_score(true_classes, predicted_classes)
    precision = precision_score(true_classes, predicted_classes, average='weighted')  # 'weighted' pode ser alterado dependendo do seu caso
    recall = recall_score(true_classes, predicted_classes, average='weighted')  # 'weighted' pode ser alterado dependendo do seu caso
    f1 = f1_score(true_classes, predicted_classes, average='weighted')  # 'weighted' pode ser alterado dependendo do seu caso

    print(f'Acurácia: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}')
    
    # Plot the confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Labels')
    plt.xlabel('Predicted Labels')
    plt.savefig('confusion_matrix_art2.png')
    plt.show()
    
    plot_history(history)
