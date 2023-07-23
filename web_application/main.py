from flask import Flask, render_template, request
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import InceptionV3, VGG16, ResNet50, MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model

app = Flask(__name__)

# Define the load_dataset function before the compare function
def load_dataset(dataset_path):
    if not os.path.exists(dataset_path):
        raise FileNotFoundError("Dataset directory does not exist: " + dataset_path)

    # Load the dataset from the directory
    return dataset_path


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/compare', methods=['POST'])
def compare():
    # Get the selected model names
    model1 = request.form.get('model1')
    model2 = request.form.get('model2')

    # Get the selected parameters
    epochs = int(request.form.get('epochs'))
    batch_size = int(request.form.get('batch_size', 16))


    # Get the form data
    dataset_link = request.form.get('dataset_link')
    dataset_path = request.form.get('dataset_path')

    if dataset_path is None:
        return "Please provide a valid dataset path."

    # Load the dataset
    dataset = load_dataset(dataset_path)

    # Preprocess the dataset based on the selected models
    preprocessed_data1, preprocessed_data2 = preprocess_data(dataset, model1, model2)

    # Train and compare the selected models
    if model1 != model2:
        history_model1, history_model2 = train_and_compare_models(model1, model2, epochs, batch_size, preprocessed_data1, preprocessed_data2)
        generate_comparison_graph(history_model1, history_model2)
    else:
        return "Please select different models for comparison."

    # Return the path to the generated comparison graph
    comparison_graph_path = os.path.join('static', 'comparison_graph.png')
    return render_template('result.html', comparison_graph_path=comparison_graph_path)


def preprocess_data(dataset, model1, model2):
    preprocessed_data1 = []
    preprocessed_data2 = []

    if model1 == 'custom-nasnetmobile':
        # Implement the preprocessing steps for Custom NASNetMobile
        preprocessed_data1 = dataset
    elif model1 == 'custom-vgg16':
        # Implement the preprocessing steps for Custom VGG16
        preprocessed_data1 = dataset
    elif model1 == 'custom-inceptionv3':
        # Implement the preprocessing steps for Custom InceptionV3
        preprocessed_data1 = dataset

    if model2 == 'custom-nasnetmobile':
        # Implement the preprocessing steps for Custom NASNetMobile
        preprocessed_data2 = dataset
    elif model2 == 'custom-vgg16':
        # Implement the preprocessing steps for Custom VGG16
        preprocessed_data2 = dataset
    elif model2 == 'custom-inceptionv3':
        # Implement the preprocessing steps for Custom InceptionV3
        preprocessed_data2 = dataset

    return preprocessed_data1, preprocessed_data2


def load_and_preprocess_dataset(dataset_path, batch_size, validation_split):
    # Create an ImageDataGenerator for data preprocessing
    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=validation_split
    )

    # Load and preprocess the training dataset
    train_data = datagen.flow_from_directory(
        dataset_path,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )

    # Load and preprocess the validation dataset
    val_data = datagen.flow_from_directory(
        dataset_path,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )

    return train_data, val_data


def build_model(model_name):
    base_model = None

    if model_name == 'custom-nasnetmobile':
        base_model = CustomNASNetMobile(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    elif model_name == 'custom-vgg16':
        base_model = CustomVGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    elif model_name == 'custom-inceptionv3':
        base_model = CustomInceptionV3(include_top=False, weights='imagenet', input_shape=(224, 224, 3))

    # Add custom classification layers on top of the base model
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(8, activation='softmax')(x)

    # Create the final model
    model = Model(inputs=base_model.input, outputs=predictions)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def train_and_compare_models(model1, model2, epochs, batch_size, data1, data2):
    # Load and preprocess the datasets
    train_data1, val_data1 = load_and_preprocess_dataset(data1, batch_size, 0.2)
    train_data2, val_data2 = load_and_preprocess_dataset(data2, batch_size, 0.2)

    # Build the models
    model1 = build_model(model1)
    model2 = build_model(model2)

    # Train the models
    history_model1 = model1.fit(train_data1, epochs=epochs, validation_data=val_data1)
    history_model2 = model2.fit(train_data2, epochs=epochs, validation_data=val_data2)

    return history_model1, history_model2


def generate_comparison_graph(history_model1, history_model2):
    # Get the training and validation accuracy
    train_acc_model1 = history_model1.history['accuracy']
    val_acc_model1 = history_model1.history['val_accuracy']
    train_acc_model2 = history_model2.history['accuracy']
    val_acc_model2 = history_model2.history['val_accuracy']

    # Get the training and validation loss
    train_loss_model1 = history_model1.history['loss']
    val_loss_model1 = history_model1.history['val_loss']
    train_loss_model2 = history_model2.history['loss']
    val_loss_model2 = history_model2.history['val_loss']

    # Plot the training and validation accuracy
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(train_acc_model1, label='Model 1 Training Accuracy')
    plt.plot(val_acc_model1, label='Model 1 Validation Accuracy')
    plt.plot(train_acc_model2, label='Model 2 Training Accuracy')
    plt.plot(val_acc_model2, label='Model 2 Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    # Plot the training and validation loss
    plt.subplot(1, 2, 2)
    plt.plot(train_loss_model1, label='Model 1 Training Loss')
    plt.plot(val_loss_model1, label='Model 1 Validation Loss')
    plt.plot(train_loss_model2, label='Model 2 Training Loss')
    plt.plot(val_loss_model2, label='Model 2 Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    # Save the comparison graph
    comparison_graph_path = os.path.join('static', 'comparison_graph.png')
    plt.savefig(comparison_graph_path)
    plt.close()

    return comparison_graph_path


if __name__ == '__main__':
    app.run(debug=True)
