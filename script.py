import os
import time
from PIL import Image
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import streamlit as st

#Ensure progress environment variable is set
os.environ["TFHUB_DOWNLOAD_PROGRESS"] = "True"

#Model path defined
SAVED_MODEL_PATH = "https://tfhub.dev/captain-pool/esrgan-tf2/1"
LOCAL_MODEL_PATH = "D:/Project/Model2"

#Create the local model directory if it doesn't exist
os.makedirs(LOCAL_MODEL_PATH, exist_ok=True)

#Load the model from TensorFlow Hub
model = hub.load(SAVED_MODEL_PATH)

def preprocess_image(image_path):
    hr_image = tf.image.decode_image(tf.io.read_file(image_path))
    if hr_image.shape[-1] == 4: 
        hr_image = hr_image[..., :-1]
    hr_size = (tf.convert_to_tensor(hr_image.shape[:-1]) // 4) * 4
    hr_image = tf.image.crop_to_bounding_box(hr_image, 0, 0, hr_size[0], hr_size[1])
    hr_image = tf.cast(hr_image, tf.float32)
    return tf.expand_dims(hr_image, 0)

def save_image(image, filename):
    if not isinstance(image, Image.Image):
        image = tf.clip_by_value(image, 0, 255)
        image = Image.fromarray(tf.cast(image, tf.uint8).numpy())
    image.save(f"{filename}.jpg")
    return f"{filename}.jpg"

def downscale_image(image):
    image_size = []
    if len(image.shape) == 4:  #Expecting batch dimension
        image_size = [image.shape[2], image.shape[1]]
    else:
        raise ValueError("Dimension mismatch. Expected batch dimension.")

    image = tf.squeeze(tf.cast(tf.clip_by_value(image, 0, 255), tf.uint8))

    # lr_image = np.asarray(
    #     Image.fromarray(image.numpy())
    #     .resize([image_size[0] // 4, image_size[1] // 4], Image.BICUBIC)
    # )

    lr_image = tf.expand_dims(lr_image, 0)
    lr_image = tf.cast(lr_image, tf.float32)
    return lr_image

#Streamlit app title
st.title("ESRGAN Image Processor")

#File uploader widget
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    #Save the uploaded input image temporarily
    temp_image_path = "temp_image.png"
    with open(temp_image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    #Preprocess the uploaded image
    hr_image = preprocess_image(temp_image_path)

    #Display the original image
    st.image(uploaded_file, caption="Original Image", use_column_width=True)
    st.write("")  # Empty line for spacing

    #Option to choose between upscaling and downscaling
    option = st.selectbox("Choose the process:", ("Upscale", "Downscale"))

    if option == "Upscale":
        #Upscale the image using the model
        start_time = time.time()
        fake_image = model(hr_image)
        fake_image = tf.squeeze(fake_image)  # Remove batch dimension
        elapsed_time = time.time() - start_time
        st.write(f"Time Taken: {elapsed_time:.2f} seconds")

        #Save and display the upscaled image
        processed_image_path = save_image(fake_image, "Super_Resolution")
        processed_image = Image.open(processed_image_path)
        st.image(processed_image, caption="Upscaled Image", use_column_width=True)

        #Option to download the processed image
        with open(processed_image_path, "rb") as f:
            st.download_button(
                label="Download Processed Image",
                data=f,
                file_name="upscaled_image.jpg",
                mime="image/jpeg"
            )
    elif option == "Downscale":
        #Downscale the image
        start_time = time.time()
        downscaled_image = downscale_image(hr_image)
        downscaled_image = tf.squeeze(downscaled_image)  #Remove batch dimension
        elapsed_time = time.time() - start_time
        st.write(f"Time Taken: {elapsed_time:.2f} seconds")

        #Save and display the downscaled image
        processed_image_path = save_image(downscaled_image, "Downscaled_Image")
        processed_image = Image.open(processed_image_path)
        st.image(processed_image, caption="Downscaled Image", use_column_width=True)

        #Option to download the processed image
        with open(processed_image_path, "rb") as f:
            st.download_button(
                label="Download Processed Image",
                data=f,
                file_name="downscaled_image.jpg",
                mime="image/jpeg"
            )