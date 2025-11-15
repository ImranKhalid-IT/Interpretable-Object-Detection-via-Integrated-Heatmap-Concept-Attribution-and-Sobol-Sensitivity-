How to Run This Code
Follow these steps to run the analysis on your own system.

1. Prerequisites
Python 3.7 or newer.

pip for installing packages.

(Recommended) An NVIDIA GPU with CUDA support. The script will automatically fall back to CPU if CUDA is not available, but it will be significantly slower.

2. Setup
Download the Code: Save the code provided as a Python file (e.g., paper_2.py).

Create a Virtual Environment (Recommended): Open your terminal, navigate to the directory where you saved the file, and run:
# Create a virtual environment
python3 -m venv venv

# Activate it
# On macOS/Linux:
source venv/bin/activate
# On Windows (cmd.exe):
.\venv\Scripts\activate
pip install torch torchvision numpy matplotlib pillow scikit-learn opencv-python requests
(Note: This installs the CPU version of PyTorch. If you have a CUDA-enabled GPU, follow the instructions on the official PyTorch website for a version compiled with GPU support.)

3. Prepare Your Image
The script is hard-coded to look for an image at /content/test5.jpg, which is a Google Colab path.

You have two simple options:

Option A (Easiest - Use Demo Image): Change line 440 from: IMAGE_PATH = '/content/test5.jpg' ...to... IMAGE_PATH = 'test5.jpg'

When you run the script, it will fail to find 'test5.jpg', and the built-in fallback will automatically download a demo image (000000039769.jpg - two cats and a frisbee) into your current directory and use that.

Option B (Use Your Own Image):

Place your own image (e.g., my_photo.jpg) in the same directory as the script.

Change line 440 to point to your file: IMAGE_PATH = 'my_photo.jpg'

4. Run the Script
With your virtual environment activated and dependencies installed, simply run the file:
python paper_2.py
Here is a README file explaining how the code works and how to run it.

Explainable Object Detection (DETR + CRAFT)
This Python script provides a comprehensive pipeline for running an object detection model (DETR) and generating two types of explanations for its predictions:

Pixel-Level Explanation: Using Grad-CAM to create a heatmap showing which pixels the model focused on for a specific detection.

Concept-Level Explanation: Using a method like CRAFT (Concept-based Rationale and Feature-based explanation) to:

Discover abstract "visual concepts" in the image using Non-negative Matrix Factorization (NMF).

Quantify the importance of each concept for a detection using Sobol scores.

The script processes a single image, detects all objects, and then generates several visualizations, culminating in a combined 3-panel plot for the primary detection.

How It Works: The Pipeline
The script executes the following steps in order:

Setup & Model Loading:

It determines whether to use a GPU (cuda) or CPU.

It initializes the ODAM_Processor, which contains a pre-defined DETRdemo (DEtection TRansformer) model with a ResNet-50 backbone.

It attempts to load pre-trained DETR weights from a public URL.

Image Processing & Detection:

It loads the target image (or downloads a demo if not found).

The image is pre-processed (resized, normalized) and fed into the DETR model.

The get_detections_features_and_input_tensor function runs the model, collects all detections (bounding boxes, class labels, and scores) above a confidence threshold (0.7).

It also saves the internal convolutional features (conv_features) from the ResNet backbone, which are crucial for the explanations.

Primary Object Selection:

From the list of all detections, it selects the one with the highest confidence score as the "primary detection."

Explanation Generation (Dual-Track):

Track A: Pixel-Level (Grad-CAM)

The generate_gradient_explanation_heatmap function is called for the primary detection.

It backpropagates the gradient from the primary object's specific logit back to the saved conv_features.

This process generates a 2D heatmap indicating pixel importance for that specific object.

Track B: Concept-Level (CRAFT/NMF)

The CRAFT_Concept_Explainer is initialized (e.g., to find 5 concepts).

fit_nmf: The saved conv_features (from all detections) are fed into an NMF model. This "learns" a dictionary of K spatial concepts (e.g., "furry texture," "wheel shape," "grassy area").

get_U_spatial: The script computes spatial activation maps for each of the K concepts, showing where each concept appears in the image.

get_sobol_scores: The script uses a SobolAnalyzer to calculate an importance score for each concept, answering: "How much did 'Concept 1' contribute to the model's detection of the 'cat'?"

Visualization:

Plot 1 (Individual Grad-CAMs): visualize_multiple_detections is called. It loops through every object detected (not just the primary one) and shows its individual Grad-CAM explanation.

Plot 2 (All Concepts): visualize_spatial_nmf_concepts is called. It shows the original image with all K learned concepts overlaid, color-coded and sorted by their Sobol importance scores.

Plot 3 (Combined Explanation): The main function visualize_combined_explanation is called. This generates the final, 3-panel summary figure for the primary detection:

Panel 1: Original Image + Bounding Box

Panel 2: Pixel-Level Explanation (Grad-CAM Overlay)

Panel 3: Concept-Level Explanation (CRAFT/NMF Concept Overlay)

🚀 How to Run This Code
Follow these steps to run the analysis on your own system.

1. Prerequisites
Python 3.7 or newer.

pip for installing packages.

(Recommended) An NVIDIA GPU with CUDA support. The script will automatically fall back to CPU if CUDA is not available, but it will be significantly slower.

2. Setup
Download the Code: Save the code provided as a Python file (e.g., paper_2.py).

Create a Virtual Environment (Recommended): Open your terminal, navigate to the directory where you saved the file, and run:

Bash

# Create a virtual environment
python3 -m venv venv

# Activate it
# On macOS/Linux:
source venv/bin/activate
# On Windows (cmd.exe):
.\venv\Scripts\activate
Install Dependencies: Install all the required libraries using pip:

Bash

pip install torch torchvision numpy matplotlib pillow scikit-learn opencv-python requests
(Note: This installs the CPU version of PyTorch. If you have a CUDA-enabled GPU, follow the instructions on the official PyTorch website for a version compiled with GPU support.)

3. Prepare Your Image
The script is hard-coded to look for an image at /content/test5.jpg, which is a Google Colab path.

You have two simple options:

Option A (Easiest - Use Demo Image): Change line 440 from: IMAGE_PATH = '/content/test5.jpg' ...to... IMAGE_PATH = 'test5.jpg'

When you run the script, it will fail to find 'test5.jpg', and the built-in fallback will automatically download a demo image (000000039769.jpg - two cats and a frisbee) into your current directory and use that.

Option B (Use Your Own Image):

Place your own image (e.g., my_photo.jpg) in the same directory as the script.

Change line 440 to point to your file: IMAGE_PATH = 'my_photo.jpg'

4. Run the Script
With your virtual environment activated and dependencies installed, simply run the file:

Bash

python paper_2.py
5. View the Output
Console Output: You will see logs in your terminal as the script progresses:

Using device: cuda
--- 1. ODAM: Initializing and processing image ---
ODAM: Found 3 objects.
ODAM: Selected primary detection: 'cat' (score: 0.99)
--- 2. ODAM: Generating gradient explanation heatmap...
--- 3. CRAFT: Initializing (Concepts: 5) ---
CRAFT: Fitting NMF to learn concept dictionary W...
...
Visualization Windows: Several Matplotlib windows will pop up on your screen.

You will first see the individual Grad-CAM plots for each of the 3 detected objects.

Next, you will see the plot showing all 5 NMF concepts.

Finally, the combined 3-panel explanation for the primary 'cat' detection will appear.

 Customization
You can easily tweak the script's behavior by changing these variables in the if __name__ == '__main__': block:

IMAGE_PATH (line 440): Set this to the path of your input image.

NUM_NMF_CONCEPTS (line 465): Change the number of concepts NMF should discover (default is 5).

conf_thresh (line 450): Change the minimum confidence score required to consider an object "detected" (default is 0.7).
