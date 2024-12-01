import streamlit as st
import cv2
import numpy as np
import pandas as pd

st.header("CIS5810 - 5 - Photography techniques")

st.header("Panning")

long_text = "Panning is a camera technique where you move the camera to follow a moving subject at a slow shutter speed, resulting in a sharp subject against a dynamically blurred background that creates a sense of motion and speed"

from streamlit.components.v1 import html

text = long_text
html(f"""
<body>
<pre style="white-space: break-spaces; font-family: sans; font-size: 1.2em">
{text}
</pre>
</body>
""")


uploaded_file = "pexels-nathansalt-2549355.jpg"

image = cv2.imread(uploaded_file)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Calculate ratio to maintain aspect ratio
scale_percent = 60  # percent of original size
width = int(image.shape[1] * scale_percent / 100)
height = int(image.shape[0] * scale_percent / 100)
dim = (width, height)

# Resize image
image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

col1_2, col2_2 = st.columns(2)

with col1_2:

    st.subheader("Panning effect")
    st.image(image)


with col2_2:
    
    st.subheader("Camera parameters")

    lst = ['Slower shutter speed', 'follow object with camera', 'Decrease aperture (large f/x)', 'Neutral density filter']
    s = ''
    for item in lst:
        s += "- " + item + "\n"
    st.markdown(s)

st.text("Image from https://www.pexels.com/photo/photo-of-man-riding-red-motor-scooter-2549355/")

st.subheader("How we can create the same effect from a photo of a motionless object?")

uploaded_file = "free-photo-of-classic-fiat-500-parked-on-a-gravel-road.png"

image = cv2.imread(uploaded_file)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Calculate ratio to maintain aspect ratio
scale_percent = 100  # percent of original size
width = int(image.shape[1] * scale_percent / 100)
height = int(image.shape[0] * scale_percent / 100)
dim = (width, height)

# Resize image
image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

st.image(image)

st.text("Image from https://www.pexels.com/photo/classic-fiat-500-parked-on-a-gravel-road-19143439/")