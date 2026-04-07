import streamlit as st
import numpy as np
import cv2
import tensorflow as tf

def dice_coefficient(y_true, y_pred):
    y_true = tf.keras.backend.flatten(y_true)
    y_pred = tf.keras.backend.flatten(y_pred)
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2 * intersection + 1) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + 1)

def combined_loss(y_true, y_pred):
    bce = tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)
    dice = dice_coefficient(y_true, y_pred)
    return bce + (1 - dice)

def iou(y_true, y_pred):
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    return intersection / (union + 1e-7)

def compute_metrics(y_true, y_pred):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()

    intersection = np.sum(y_true * y_pred)

    dice = (2 * intersection + 1) / (np.sum(y_true) + np.sum(y_pred) + 1)
    iou = intersection / (np.sum(y_true) + np.sum(y_pred) - intersection + 1e-7)

    return dice, iou

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "final_model.keras",
        custom_objects={
            "combined_loss": combined_loss,
            "dice_coefficient": dice_coefficient,
            "iou": iou
        }
    )

model = load_model()

st.title("🎬 Scene Cast AI - Face Segmentation")

st.markdown("---")
st.subheader("🎯 Model Info")

st.write(
"""
- Architecture: U-Net (MobileNetV2 Encoder)
- Input Size: 256×256
- Task: Face Segmentation
""")

st.divider()

# Display the metrics
st.subheader("📊 Overall Performance Metrics")

col1, col2 = st.columns(2)

col1.metric("Dice Score", "0.67")
col2.metric("IoU Score", "0.50")

st.divider()

st.markdown("The model loaded!")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

def preprocess(image):
    img = cv2.resize(image, (256,256))
    img = img / 255.0
    return img

def predict(image):
    img = preprocess(image)
    # pred = model.predict(np.expand_dims(img, axis=0))[0]
    # pred = model.predict(img)[0]
    # mask = (pred > 0.5).astype(np.uint8)
    # return mask
    #img = cv2.resize(image, (224, 224))
    #img = img / 255.0
    img = np.expand_dims(img, axis=0)  # add batch dimension
    
    print("Input shape:", img.shape)  # debug
    
    pred = model.predict(img)[0]
    return pred

def overlay(image, mask):
    mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
    overlay = image.copy()
    # overlay[mask[:,:,0] == 1] = [0,255,0]
    overlay[mask == 1] = [0,255,0]
    return overlay

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    pred = predict(image)
    mask = (pred > 0.5).astype(np.uint8)
    result = overlay(image, mask)

    st.image(image, caption="Original")
    st.image(pred.squeeze(), caption="Predicted Mask")
    st.image(result, caption="Overlay Output")
