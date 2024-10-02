import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pandas as pd


st.header("CIS5810 - 5 - Photography techniques")

st.header("Panning using kernel motion blur")


def apply_panning_effect(img, blur_amount=15, subject_area=None):
    # Create the motion blur kernel
    kernel_motion_blur = np.zeros((blur_amount, blur_amount))
    kernel_motion_blur[int((blur_amount-1)/2), :] = np.ones(blur_amount)
    kernel_motion_blur = kernel_motion_blur / blur_amount
    
    # Apply the motion blur
    blurred = cv2.filter2D(img, -1, kernel_motion_blur)
    
    # If a subject area is specified, keep it sharp
    if subject_area:
        x, y, w, h = subject_area
        blurred[y:y+h, x:x+w] = img[y:y+h, x:x+w]
    
    return blurred

uploaded_file = "free-photo-of-classic-fiat-500-parked-on-a-gravel-road.png"

def crop_image(image):
    # Define the coordinates for the 10x10 pixel segment (e.g., top-left corner at (x, y))
    x, y = 50, 50  # Example coordinates

    segment = image[y:y+10, x:x+10]

    # Resize the segment to 200x200 pixels
    resized_segment = cv2.resize(segment, (200, 200), interpolation=cv2.INTER_NEAREST)

    # Convert color from BGR to RGB for display
    resized_segment_rgb = cv2.cvtColor(resized_segment, cv2.COLOR_BGR2RGB)

    return resized_segment_rgb

if uploaded_file is not None:
    image = cv2.imread(uploaded_file)
    img_array = np.array(image)

    # Create two columns for side-by-side display
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(image)

    subject_area = (60, 200, 200, 130)
    blur_amount = st.slider("Blur Amount", min_value=1, max_value=50, value=1)

    result = apply_panning_effect(img_array, blur_amount, subject_area)

    with col2:
        st.subheader("Result")
        st.image(result)
    

    # Create two columns for side-by-side display
    col1_2, col2_2 = st.columns(2)
    y,x = 50, 50

    with col1_2:
        st.subheader("10x10 original Image")
        crop_image = crop_image(image)
        st.image(crop_image)

    with col2_2:
        st.subheader("10x10 Result")
        
        segment = result[y:y+10, x:x+10]

        # Resize the segment to 200x200 pixels
        resized_segment = cv2.resize(segment, (200, 200), interpolation=cv2.INTER_NEAREST)

        # Convert color from BGR to RGB for display
        crop_result = cv2.cvtColor(resized_segment, cv2.COLOR_BGR2RGB)
        
        # crop_result = crop_image(result)
        st.image(crop_result, width = 200)

    st.subheader("Kernel Motion Blur")
    
    kernel_motion_blur = np.zeros((blur_amount, blur_amount))
    kernel_motion_blur[int((blur_amount-1)/2), :] = np.ones(blur_amount)
    kernel_motion_blur = kernel_motion_blur / blur_amount
    
    # Convert the kernel to a DataFrame for display
    df_kernel = pd.DataFrame(kernel_motion_blur, columns=[f"Col {i+1}" for i in range(blur_amount)])

    # Use Pandas Styler to hide the index and headers
    styled_df = df_kernel.style.hide(axis='index').hide(axis='columns').set_table_styles(
        [{'selector': 'td', 'props': [('border', '1px solid black')]}]
    )

    # Render the styled DataFrame as HTML in Streamlit
    st.markdown(styled_df.to_html(), unsafe_allow_html=True)

st.text("Image from https://www.pexels.com/photo/classic-fiat-500-parked-on-a-gravel-road-19143439/")
        