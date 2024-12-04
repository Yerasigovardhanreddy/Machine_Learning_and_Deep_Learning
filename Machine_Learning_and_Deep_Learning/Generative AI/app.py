import streamlit as st
from PIL import Image
from ultralytics import YOLO  # YOLO for object detection
from transformers import BlipProcessor, BlipForConditionalGeneration
from gtts import gTTS
import cv2
import numpy as np
import os

# Title with confidentiality and description
st.title("Generative AI for Object Recognition")
st.markdown("""*Key Features:*  
1. *Upload an image* securely.  
2. Get a *detailed description* of the image.
3. Perform *object detection* with bounding boxes.
4. Hear the description as *audio output*! 🎧  
5. Receive *safety and situational insights* based on detected objects.  
""")
# File uploader
uploaded_file = st.file_uploader("Upload an image file (JPEG, PNG)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and convert the uploaded image to RGB mode
    image = Image.open(uploaded_file)
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # Display the uploaded image
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Image description section
    st.header("📝 Image Description:")
    # Load BLIP model for image captioning
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    # Preprocess and generate image description
    inputs = processor(images=image, return_tensors="pt")
    out = model.generate(**inputs)
    description = processor.decode(out[0], skip_special_tokens=True)

    # Display the description
    st.success(description)

    # Audio generation section
    st.header("🔊 Audio Description:")
    audio_file_path = "description.mp3"
    tts = gTTS(description, lang='en')
    tts.save(audio_file_path)

    # Play the audio
    st.audio(audio_file_path, format="audio/mp3")

    # YOLO object detection section
    st.header("📦 Object Detection:")
    yolo_model = YOLO("yolov8n.pt")  # Load YOLO model (Nano version for speed)
    
    # Save the uploaded image locally for YOLO processing
    uploaded_image_path = "uploaded_image.jpg"
    image.save(uploaded_image_path)

    # Perform object detection
    results = yolo_model(uploaded_image_path)
    detected_results = results[0]  # Get the first result

    # Annotated image with bounding boxes
    annotated_image = detected_results.plot()

    # Convert NumPy array (annotated_image) to Pillow Image for Streamlit
    annotated_image_pil = Image.fromarray(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))

    # Display annotated image with bounding boxes
    st.image(annotated_image_pil, caption="Object Detection Results", use_container_width=True)

    # Extract detected object names
    st.header("🛡 Safety Insights:")
    detected_objects = detected_results.boxes.cls.cpu().numpy()
    detected_names = [detected_results.names[int(cls)] for cls in detected_objects]

    # Ensure safety tips are displayed only once per object type
    displayed_objects = set()
    if detected_names:
        st.write("The following objects were detected in the image:")
        for obj_name in set(detected_names):  # Use set to remove duplicates
            st.write(f"- *{obj_name}*")
            if obj_name not in displayed_objects:
                if obj_name in ["car", "truck", "bus"]:
                    st.info(f"Detected a *{obj_name}*. Ensure you maintain a safe distance and avoid distractions.")
                elif obj_name == "person":
                    st.info("Detected a *person*. Stay vigilant and avoid overcrowded areas if possible.")
                elif obj_name in ["bicycle", "motorcycle"]:
                    st.info(f"Detected a *{obj_name}*. Be cautious of smaller vehicles on the road.")
                elif obj_name in ["dog", "cat"]:
                    st.info(f"Detected a *{obj_name}*. Ensure the area is safe for pets.")
                else:
                    st.info(f"Detected a *{obj_name}*. Stay aware and cautious.")
                displayed_objects.add(obj_name)
    else:
        st.warning("No objects detected in the image.")

    # Cleanup temporary files
    os.remove(uploaded_image_path)
    os.remove(audio_file_path)
else:
    st.info("Please upload an image to proceed.")