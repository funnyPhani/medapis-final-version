from dotenv import load_dotenv
import streamlit as st
import os
import google.generativeai as genai
from PIL import Image
import requests
from io import BytesIO

# Load environment variables
load_dotenv()

# Configure the "genai" library by providing API key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def upload_file(file_path, mime_type):
    # Upload the file using the File API with the correct MIME type
    file = genai.upload_file(file_path, mime_type=mime_type)
    return file

def get_gemini_response(input_text, file_ref):
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content([input_text, file_ref])
    return response.text

def input_image_setup(uploaded_file):
    if uploaded_file is not None:
        # Save the uploaded file to a temporary path
        file_extension = uploaded_file.name.split('.')[-1]
        temp_image_path = f"temp_image.{file_extension}"
        with open(temp_image_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        return temp_image_path, uploaded_file.type
    else:
        raise FileNotFoundError("No file uploaded")

def process_image_file(image_file):
    if image_file.type.startswith('image'):
        return image_file
    else:
        st.error("Please upload an image file.")

def process_image_url(image_url):
    try:
        response = requests.get(image_url)
        if response.status_code == 200:
            image_data = Image.open(BytesIO(response.content))
            # Save image from URL to a temporary path
            image_path = "temp_image_from_url.jpg"
            image_data.save(image_path)
            return image_path, "image/jpeg"
        else:
            st.error("Failed to fetch image from URL. Please make sure the URL is correct.")
    except Exception as e:
        st.error(f"Error: {e}")

# Initialize the Streamlit app
st.set_page_config(page_title="WellnessAI Advisor")
st.header("WellnessAI Advisor 👨‍⚕️🧑‍⚕️")

# Initialize variables
uploaded_file = None
image_path = None
mime_type = None

# Image source selection
image_source = st.radio("Select image source:", ("Local", "URL"))

if image_source == "Local":
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image_path, mime_type = input_image_setup(uploaded_file)
        image = Image.open(image_path)
        st.image(image, caption="Uploaded Image.", use_column_width=True)
elif image_source == "URL":
    image_url = st.text_input("Enter Image URL")
    if image_url:
        image_path, mime_type = process_image_url(image_url)
        if image_path:
            image = Image.open(image_path)
            st.image(image, caption="Image from URL.", use_column_width=True)

# Buttons for selecting functionality
medical_button = st.button("Analyze Medical Image")
calorie_button = st.button("Analyze Calorie Intake")

input_prompt_calorie =  """
    You are an expert in nutritionist where you need to see the food items
    and calculate the total calories, also provide the details of every food items with calories intake
    is below format
    1. Item 1 - no of calories
    2. Item 2 - no of calories
    ----
    ----
    After that mention that the meal is healthy meal or not and also mention the percentage split of ratio of
    carbohydrates,proteins, fats, sugar and calories in meal.
    finally give suggestion which item should me removed and which items should be added it meal to make the
    meal healthy if it's unhealthy. 
    """

input_prompt_medical = """
You are a medical practitioner and an expert in analyzing medical-related images working for a very reputed hospital. You will be provided with images and you need to identify anomalies, any disease, or health issues. You need to generate the result in a detailed manner. Write all the findings, next steps, recommendations, etc. You only need to respond if the image is related to a human body and health issues. You must answer but also write a disclaimer saying, "Consult with a Doctor before making any decisions."

Remember, if certain aspects are not clear from the image, it's okay to state 'Unable to determine based on the provided image.'
"""

# If medical button is clicked
if medical_button:
    if image_path:
        file_ref = upload_file(image_path, mime_type)
        response = get_gemini_response(input_prompt_medical, file_ref)
        st.subheader("Medical Analysis Response")
        st.write(response)
    else:
        st.error("Please upload an image or provide an image URL.")

# If calorie button is clicked
if calorie_button:
    if image_path:
        file_ref = upload_file(image_path, mime_type)
        response = get_gemini_response(input_prompt_calorie, file_ref)
        st.subheader("Calorie Analysis Response")
        st.write(response)
    else:
        st.error("Please upload an image or provide an image URL.")