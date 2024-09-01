# ESRGAN Image Processor

This repository provides an implementation of an image processing application using the ESRGAN model. The application can upscale or downscale images with a user-friendly interface implemented using Streamlit.

## SRGAN

![SRGAN](https://image.slidesharecdn.com/srgan-190910075433/75/Photo-realistic-Single-Image-Super-resolution-using-a-Generative-Adversarial-Network-SRGAN-20-2048.jpg)

## ESRGAN

![ESRGAN](https://miro.medium.com/v2/resize:fit:1400/1*QEi9afEvGmsXLfWJTi-4Aw.png)

#### Overview
ESRGAN (Enhanced Super-Resolution Generative Adversarial Network) builds upon the SRGAN framework to further improve image resolution and quality. It utilizes advanced network components and loss functions to generate high-resolution images with finer details and better texture representation.

#### Architecture
ESRGAN features an enhanced Generator and Discriminator architecture. It replaces the standard residual blocks with Residual-in-Residual Dense Blocks (RRDB) and removes batch normalization layers. This advanced structure aids in better feature representation and stable training.

#### Key Features
- **Generator**: Utilizes RRDB, which integrates residual and dense connections to capture and reconstruct fine image details effectively.
- **Discriminator**: Employs a relativistic adversarial approach, helping the Generator produce more realistic images by better estimating image quality.
- **Loss Functions**: Combines a sophisticated perceptual loss, calculated through a pre-trained VGG network, and a relativistic adversarial loss to enhance texture details and visual realism.

#### Applications
ESRGAN is widely used in areas that require high-quality image upscaling such as medical imaging, satellite imagery, and enhancing low-resolution photos and videos. Its ability to produce detailed and realistic images makes it suitable for professional and consumer applications alike.


## Features

- Upload an image file in JPG, PNG, or JPEG format.
- Select whether to upscale or downscale the image.

## Requirements

- Python 3.x
- TensorFlow
- TensorFlow Hub
- NumPy
- Pillow
- Streamlit

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-repo/ESRGAN-Image-Processor.git
   cd ESRGAN-Image-Processor
   ```

2. Set up a virtual environment (recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

### `requirements.txt`

Ensure the following packages are listed in your `requirements.txt` file:

```text
tensorflow
tensorflow_hub
numpy
Pillow
streamlit
```

Required package versions:

```text
tensorflow==2.5.0
tensorflow_hub==0.12.0
numpy==1.19.5
Pillow==8.2.0
streamlit==0.82.0
```

## Usage

1. Run the Streamlit app:

   ```bash
   streamlit run app.py
   ```

2. Open your web browser and navigate to `http://localhost:8501`.

3. Upload an image and choose whether to upscale or downscale the image. The application will process the image and provide an option to download the result.

### Code Overview

- The application first ensures that the TensorFlow Hub download progress is set.
- It defines the path for the ESRGAN model, checks for or creates the directory for the local copy, and loads the model using TensorFlow Hub.
- The `preprocess_image` function loads and preprocesses the image.
- The `save_image` function saves the processed image.
- The `downscale_image` function downscales the image using TensorFlow operations.
- The Streamlit app provides the user interface to upload images, choose processing options, and display/download the processed images.

