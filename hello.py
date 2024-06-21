import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os


def load_and_preprocess_image(uploaded_file, img_size):
    image = Image.open(uploaded_file).convert('RGB')
    image = image.resize(img_size)
    image = np.array(image)
    image = image / 255.0  
    image = np.expand_dims(image, axis=0)  
    return image


def visualize_color_distribution(image):
    color_distribution = np.bincount(image.flatten(), minlength=256)

    x_values = np.arange(256)
    y_values = color_distribution

    
    plt.figure(figsize=(10, 6))
    plt.scatter(x_values, y_values, s=10, color='blue', alpha=0.7)
    plt.title('Color Distribution (Scatter Plot)')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Count')
    st.pyplot(plt)


def visualize_dataset_images(train_ds, class_names):
    plt.figure(figsize=(10, 10))
    for images, labels in train_ds.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype('uint8'))
            plt.title(class_names[labels[i]])
            plt.axis('off')
    st.pyplot(plt)


def classify_uploaded_image(image, img_size):
  
    model = tf.keras.applications.MobileNetV2(weights='imagenet', input_shape=(224, 224, 3))

    image = load_and_preprocess_image(image, img_size)

    predictions = model.predict(image)

   
    decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=3)[0]

    return decoded_predictions

# Streamlit
st.title("Image Analysis Application")


img_size = (224, 224)


uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
    st.write("")

    if st.button("Analyse Image"):
        st.write("Analyzing...")

       
        image = load_and_preprocess_image(uploaded_file, img_size)

       
        model = tf.keras.applications.MobileNetV2(weights='imagenet', input_shape=(224, 224, 3))

       
        img = np.array(Image.open(uploaded_file).convert('L'))  # Convert to grayscale for simplicity
        visualize_color_distribution(img)
        
        predictions = model.predict(image)

        decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=3)[0]

        st.write("Prediction Results:")
        for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
            st.write(f"{i+1}. {label}: {score:.2f}")

       


data_dir = r"C:\Users\sehja\Downloads\untitled Folder2\images"
batch_size = 32


if os.path.isdir(data_dir):
   
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.3, 
        subset='training',
        seed=123,
        image_size=img_size,
        batch_size=batch_size,
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.3,
        subset='validation',
        seed=123,
        image_size=img_size,
        batch_size=batch_size,
    )

   
    class_names = train_ds.class_names
    st.write("Class Names in the Dataset:")
    st.write(class_names)

    
    st.write("Sample Images from Dataset:")
    visualize_dataset_images(train_ds, class_names)
else:
    st.warning("Please ensure the dataset directory path is correct.")

