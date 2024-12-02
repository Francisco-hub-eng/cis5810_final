import streamlit as st
import cv2
import numpy as np
import pandas as pd
from streamlit.components.v1 import html



st.header("CIS5810 - 5 - Photography techniques")

option = st.selectbox(
    "Segmentation technique:",
    ["Felzenswald", "Clustering", "Edge", "Region", "Neural network"],
    index=0,
    placeholder="Choose an option"
)

if option == "Felzenswald":

    st.subheader("Felzenszwald")

    from skimage.segmentation import felzenszwalb


    long_text = "Felzenszwalb segmentation is a graph-based image segmentation algorithm that partitions images into meaningful regions, and represent the image as an undirected graph. The idea is that would recognize the car as a maningful region, but recongize different segments of the car as separate groups. One strategy could be to after the process, select all the segments that belong to the car by eye. Another problem is that some background of the car is similar to the car"


    text = long_text
    html(f"""
    <body>
    <pre style="white-space: break-spaces; font-family: sans; font-size: 1.2em">
    {text}
    </pre>
    </body>
    """)

    # Load the image
    image = cv2.imread('free-photo-of-classic-fiat-500-parked-on-a-gravel-road.jpeg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    k_value = st.select_slider(
        'Select scale (k) parameter for Felzenszwalb segmentation',
        options=np.arange(0, 3000, 100),  # Values from 100 to 1000
        value=1500  # Default value
    )

    # Apply Felzenszwalb segmentation
    segments = felzenszwalb(image, scale=k_value, sigma=0.7, min_size=10)

    # Create a mask for the car region
    # The car is likely to be one of the larger segments in bottom right
    unique_segments = np.unique(segments)
    car_mask = np.zeros_like(segments, dtype=bool)

    # Find the segment corresponding to the car
    # Usually the car segment will be one of the larger segments in bottom right
    h, w = segments.shape
    #center_segment = segments[h//2, w//2]
    center_segment = segments[510, 250]

    car_mask = segments == center_segment

    # Apply the mask to the original image
    result = image.copy()
    result[~car_mask] = 0

    # Optional: Clean up the mask using morphological operations
    kernel = np.ones((5,5), np.uint8)
    car_mask = cv2.morphologyEx(car_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    car_mask = cv2.morphologyEx(car_mask, cv2.MORPH_OPEN, kernel)


    col1, col2, col3, col4 = st.columns(4)

    with col1:
        
        # Draw the X
        length = 30  # Length of each line of the X
        x, y = 250, 510  # Your coordinates
        # Draw the two diagonal lines that form the X
        cv2.line(image, (x - length, y - length), (x + length, y + length), (255, 0, 0), 5)  # red color, thickness 3
        cv2.line(image, (x + length, y - length), (x - length, y + length), (255,0, 0), 5)

        st.subheader("Image")
        st.image(image)


    with col2:
        st.subheader("Segments")
        normalized = cv2.normalize(segments, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        st.image(normalized)

    with col3:
        st.subheader("mask")
        car_mask = cv2.normalize(car_mask, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        st.image(car_mask)

    with col4:
        st.subheader("Merge")
        st.image(result)

if option == "Clustering":


    st.subheader("Clustering-based segmentation")

    long_text = "This algorithm cluster pixel by common properties. Sadly in this case clusters some parts of the car with the floor. Car properties are similar to the floor and sky, so I can not use it as it is. I tried with different k values, better results are with k=2 or k=3"


    text = long_text
    html(f"""
    <body>
    <pre style="white-space: break-spaces; font-family: sans; font-size: 1.2em">
    {text}
    </pre>
    </body>
    """)

    # Load the image
    img = cv2.imread('free-photo-of-classic-fiat-500-parked-on-a-gravel-road.jpeg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    import distinctipy

    vectorized = img.reshape((-1,3))
    vectorized = np.float32(vectorized)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    k_value = st.select_slider(
        'Select number of clusters (k)',
        options=np.arange(1, 50, 1),  # Values from 100 to 1000
        value=3 # Default value
    )

    K = k_value
    attempts=10
    ret,label,center=cv2.kmeans(vectorized,K,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)

    center = np.uint8(center)

    res = center[label.flatten()]
    result_image = res.reshape((img.shape))

    # Generate visually distinct colors using distinctipy
    colors = distinctipy.get_colors(K)
    # Convert colors from 0-1 range to 0-255 range
    colors = [(int(r*255), int(g*255), int(b*255)) for r,g,b in colors]

    # Create color-coded segmentation
    segmented = np.zeros_like(vectorized)
    for i in range(K):
        segmented[label.flatten() == i] = colors[i]

    # Reshape back to original image dimensions
    result_image = segmented.reshape(img.shape)

    # create mask
    target_value = result_image[510, 250]
    mask = cv2.inRange(result_image, target_value, target_value)  

    # result
    result = cv2.bitwise_and(img, img, mask=mask)
    result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)


    col1, col2, col3, col4 = st.columns(4)

    with col1:

        # Draw the X
        length = 30  # Length of each line of the X
        x, y = 250, 510  # Your coordinates
        # Draw the two diagonal lines that form the X
        cv2.line(img, (x - length, y - length), (x + length, y + length), (255, 0, 0), 5)  # red color, thickness 3
        cv2.line(img, (x + length, y - length), (x - length, y + length), (255,0, 0), 5)

        st.subheader("Image")
        st.image(img)

    with col2:
        st.subheader("Segments")
        normalized = cv2.normalize(result_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        st.image(normalized)

    with col3:
        st.subheader("mask")
        st.image(mask)

    with col4:
        st.subheader("merge")
        normalized = cv2.normalize(result_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        st.image(result)

if option == "Edge":

    st.subheader("Edge detection segmentation")

    from scipy import ndimage
    from skimage.color import rgb2gray

    long_text = "Edge detection aids in segmentation by identifying boundaries between regions based on abrupt changes in pixel intensity or color, which helps delineate distinct objects or areas within an image."


    text = long_text
    html(f"""
    <body>
    <pre style="white-space: break-spaces; font-family: sans; font-size: 1.2em">
    {text}
    </pre>
    </body>
    """)

    # Load the image
    img = cv2.imread('free-photo-of-classic-fiat-500-parked-on-a-gravel-road.jpeg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    gray = rgb2gray(img)

    sobel_horizontal = np.array([np.array([1, 2, 1]), np.array([0, 0, 0]), np.array([-1, -2, -1])])
    sobel_vertical = np.array([np.array([-1, 0, 1]), np.array([-2, 0, 2]), np.array([-1, 0, 1])])

    out_h = ndimage.convolve(gray, sobel_horizontal, mode='reflect')
    out_v = ndimage.convolve(gray, sobel_vertical, mode='reflect')
    out = out_h + out_v

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.subheader("Image")
        st.image(img)

    with col2:
        st.subheader("horizontal")
        normalized = cv2.normalize(out_h, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        st.image(normalized)

    with col3:
        st.subheader("vertical")
        normalized = cv2.normalize(out_v, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        st.image(normalized)

    with col4:
        st.subheader("both")
        normalized = cv2.normalize(out, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        st.image(normalized)

if option == "Region":

    st.subheader("Region-based segmentation")

    from scipy import ndimage
    from skimage.color import rgb2gray

    long_text = "Region-based segmentation groups pixels into regions"

    text = long_text
    html(f"""
    <body>
    <pre style="white-space: break-spaces; font-family: sans; font-size: 1.2em">
    {text}
    </pre>
    </body>
    """)

    # Load the image
    img = cv2.imread('free-photo-of-classic-fiat-500-parked-on-a-gravel-road.jpeg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


    gray = rgb2gray(img)


    gray_r = gray.reshape(gray.shape[0]*gray.shape[1])
    for i in range(gray_r.shape[0]):
        if gray_r[i] > gray_r.mean():
            gray_r[i] = 1
        else:
            gray_r[i] = 0
    gray = gray_r.reshape(gray.shape[0],gray.shape[1])


    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Image")
        st.image(img)

    with col2:
        st.subheader("gray")
        normalized = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        st.image(normalized)

if option == "Neural network":

    st.subheader("Segmentation with neural networks - rembg")

    from rembg import remove
    from PIL import Image

    long_text = "Remove image backgrounds using deep neural network from the rembg python library"

    text = long_text
    html(f"""
    <body>
    <pre style="white-space: break-spaces; font-family: sans; font-size: 1.2em">
    {text}
    </pre>
    </body>
    """)

    # Load the image
    img = cv2.imread('free-photo-of-classic-fiat-500-parked-on-a-gravel-road.jpeg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    from rembg import remove
    from PIL import Image

    output = remove(img)

    # Read the image
    mask = cv2.imread('no_background.png')

    # Create binary mask using threshold
    binary_mask = cv2.threshold(mask, 0, 1, cv2.THRESH_BINARY)[1]

    binary_mask = (binary_mask * 255).astype(np.uint8)


    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.subheader("Image")
        st.image(img)

    with col2:
        st.subheader("object")
        normalized = cv2.normalize(output, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        st.image(normalized)

    with col3:
        st.subheader("mask")
        normalized = cv2.normalize(binary_mask, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        st.image(normalized)

    with col4:
        st.subheader("result")
        normalized = cv2.normalize(img * binary_mask, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        st.image(normalized)

    st.write("Rembg library obtained from: https://github.com/danielgatis/rembg")

