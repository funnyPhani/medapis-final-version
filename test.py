# # import ollama
# # from rich import print
# # res = ollama.chat(
# # 	model="llava",
# # 	messages=[
# # 		{
# # 			'role': 'user',
# # 			'content': f"""
# #                     You are a medical practitioner and an expert in analyzing medical images. 
# #                     Identify any anomalies, diseases, or health issues in the image. 
# #                     Provide detailed findings, recommendations, and next steps. 
# #                     If certain aspects are unclear, state 'Unable to determine based on the provided image.' 
# #                     Also, include a disclaimer: 'Consult with a doctor before making any decisions.'
# #                     Provide the information in only markdown format.
# #                        """,
# # 			'images': ['./1800ss_beforeitsnews_rf_morgellons_disease.jpg']
# # 		}
# # 	]
# # )

# # print(res['message']['content'])


# import ollama
# from rich import print

# # Open the image in binary mode
# with open('./1800ss_beforeitsnews_rf_morgellons_disease.jpg', 'rb') as image_file:
#     image_bytes = image_file.read()

# # Send the image bytes in the request
# res = ollama.chat(
#     model="llava:latest",
#     messages=[
#         {
#             'role': 'user',
#             'content': """
#                 You are a medical practitioner and an expert in analyzing medical images. 
#                 Identify any anomalies, diseases, or health issues in the image. 
#                 Provide detailed findings, recommendations, and next steps. 
#                 If certain aspects are unclear, state 'Unable to determine based on the provided image.' 
#                 Also, include a disclaimer: 'Consult with a doctor before making any decisions.'
#                 Provide the information in only markdown format.
#             """,
#             'images': [image_bytes]
#         }
#     ]
# )

# # Print the result
# print(res['message']['content'])

# import ollama
# from rich import print

# # Open the image in binary mode
# with open('./thali-indian-1296x728-header.jpg', 'rb') as image_file:
#     image_bytes = image_file.read()

# # Send the image bytes in the request
# res = ollama.chat(
#     model="llava:latest",
#     messages=[
#         {
#             'role': 'user',
#             'content': """
#                         You are an expert in nutritionist where you need to see the food items
#                         and calculate the total calories, also provide the details of every food items with calories intake
#                         is below format
#                         1. Item 1 - no of calories
#                         2. Item 2 - no of calories
#                         ----
#                         ----
#                         After that mention that the meal is healthy meal or not and also mention the percentage split of ratio of
#                         carbohydrates,proteins, fats, sugar and calories in meal.
#                         finally give suggestion which item should me removed and which items should be added it meal to make the
#                         meal healthy if it's unhealthy.
#                     """,
#             'images': [image_bytes]
#         }
#     ]
# )

# # Print the result
# print(res['message']['content'])
                    






# # from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
# # from qwen_vl_utils import process_vision_info

# # # default: Load the model on the available device(s)
# # model = Qwen2VLForConditionalGeneration.from_pretrained(
# #     "Qwen/Qwen2-VL-2B-Instruct", torch_dtype="auto", device_map="auto"
# # )

# # # We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
# # # model = Qwen2VLForConditionalGeneration.from_pretrained(
# # #     "Qwen/Qwen2-VL-2B-Instruct",
# # #     torch_dtype=torch.bfloat16,
# # #     attn_implementation="flash_attention_2",
# # #     device_map="auto",
# # # )

# # # default processer
# # processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")

# # # The default range for the number of visual tokens per image in the model is 4-16384. You can set min_pixels and max_pixels according to your needs, such as a token count range of 256-1280, to balance speed and memory usage.
# # # min_pixels = 256*28*28
# # # max_pixels = 1280*28*28
# # # processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)

# # messages = [
# #     {
# #         "role": "user",
# #         "content": [
# #             {
# #                 "type": "image",
# #                 "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
# #             },
# #             {"type": "text", "text": "Describe this image."},
# #         ],
# #     }
# # ]

# # # Preparation for inference
# # text = processor.apply_chat_template(
# #     messages, tokenize=False, add_generation_prompt=True
# # )
# # image_inputs, video_inputs = process_vision_info(messages)
# # inputs = processor(
# #     text=[text],
# #     images=image_inputs,
# #     videos=video_inputs,
# #     padding=True,
# #     return_tensors="pt",
# # )
# # inputs = inputs.to("cuda")

# # # Inference: Generation of the output
# # generated_ids = model.generate(**inputs, max_new_tokens=128)
# # generated_ids_trimmed = [
# #     out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
# # ]
# # output_text = processor.batch_decode(
# #     generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
# # )
# # print(output_text)

































# # # import fitz  # PyMuPDF

# # # def crop_pdf(input_pdf, output_pdf, crop_rect):
# # #     # Open the input PDF
# # #     pdf_document = fitz.open(input_pdf)

# # #     # Iterate through all pages
# # #     for page_num in range(pdf_document.page_count):
# # #         page = pdf_document.load_page(page_num)

# # #         # Get the current page's rectangle (bounding box)
# # #         page_rect = page.rect
        
# # #         # Define the new crop rectangle (top-left, bottom-right corners)
# # #         # crop_rect should be in the format (x0, y0, x1, y1)
# # #         x0, y0, x1, y1 = crop_rect
        
# # #         # Ensure the crop rectangle is within page bounds
# # #         new_rect = fitz.Rect(max(page_rect.x0, x0),
# # #                              max(page_rect.y0, y0),
# # #                              min(page_rect.x1, x1),
# # #                              min(page_rect.y1, y1))
        
# # #         # Apply the crop to the page
# # #         page.set_cropbox(new_rect)
    
# # #     # Save the cropped PDF
# # #     pdf_document.save(output_pdf)
# # #     pdf_document.close()

# # # # Define the input/output PDFs and the crop rectangle (in points)
# # # input_pdf_path = r"C:\Users\A507658\Downloads\med_os_apis\vggfpn.pdf"
# # # output_pdf_path = "output_cropped.pdf"
# # # crop_rectangle = (50, 50, 400, 600)  # Adjust coordinates based on your crop needs

# # # # Call the function to crop the PDF
# # # crop_pdf(input_pdf_path, output_pdf_path, crop_rectangle)

# # # print(f"Cropped PDF saved as {output_pdf_path}")
# # import fitz  # PyMuPDF

# # def crop_upper_and_bottom(input_pdf, output_pdf, top_crop, bottom_crop):
# #     # Open the input PDF
# #     pdf_document = fitz.open(input_pdf)

# #     # Iterate through all pages
# #     for page_num in range(pdf_document.page_count):
# #         page = pdf_document.load_page(page_num)

# #         # Get the current page's rectangle (bounding box)
# #         page_rect = page.rect
        
# #         # Define the new crop rectangle by reducing height (top and bottom)
# #         # Reduce the top by `top_crop` and bottom by `bottom_crop`
# #         new_rect = fitz.Rect(
# #             page_rect.x0,               # Left boundary (unchanged)
# #             page_rect.y0 + top_crop,     # Top boundary (cropped by `top_crop`)
# #             page_rect.x1,               # Right boundary (unchanged)
# #             page_rect.y1 - bottom_crop   # Bottom boundary (cropped by `bottom_crop`)
# #         )
        
# #         # Apply the crop to the page
# #         page.set_cropbox(new_rect)
    
# #     # Save the cropped PDF
# #     pdf_document.save(output_pdf)
# #     pdf_document.close()

# # # Define the input/output PDFs and the crop values
# # input_pdf_path = r"C:\Users\A507658\Downloads\med_os_apis\vggfpn.pdf"
# # output_pdf_path = "output_cropped.pdf"
# # top_crop = 123  # Crop 50 points from the top
# # bottom_crop = 123  # Crop 50 points from the bottom

# # # Call the function to crop the PDF
# # crop_upper_and_bottom(input_pdf_path, output_pdf_path, top_crop, bottom_crop)

# # print(f"Cropped PDF saved as {output_pdf_path}")





# from fastapi import FastAPI, File, UploadFile, Form
# from typing import Optional
# import ollama
# from fastapi.responses import HTMLResponse
# import requests

# app = FastAPI()

# # Helper function to fetch image from URL
# def fetch_image_from_url(image_url: str):
#     response = requests.get(image_url)
#     if response.status_code == 200:
#         return response.content
#     return None



# # Endpoint 1: Analyze medical image (file or URL)
# @app.post("/analyze-medical-image/")
# async def analyze_medical_image(file: UploadFile = File(None), image_url: Optional[str] = Form(None)):
#     image_bytes = None
    
#     if file:
#         image_bytes = await file.read()  # Read the uploaded file
#     elif image_url:
#         image_bytes = fetch_image_from_url(image_url)  # Fetch image from URL
    
#     if image_bytes is None:
#         return {"error": "No valid image provided."}

#     # Call the ollama model with the image
#     res = ollama.chat(
#         model="llava:latest",
#         messages=[
#             {
#                 'role': 'user',
#                 'content': """
#                     You are a medical practitioner and an expert in analyzing medical images. 
#                     Identify any anomalies, diseases, or health issues in the image. 
#                     Provide detailed findings, recommendations, and next steps. 
#                     If certain aspects are unclear, state 'Unable to determine based on the provided image.' 
#                     Also, include a disclaimer: 'Consult with a doctor before making any decisions.'
#                     Provide the information in only markdown format.
#                 """,
#                 'images': [image_bytes]
#             }
#         ]
#     )
#     print("-"*100)
#     print({"result": res['message']['content']})
#     print("-"*100)

#     return {"result": res['message']['content']}

# # Endpoint 2: Analyze food image for nutrition (file or URL)
# @app.post("/analyze-food-image/")
# async def analyze_food_image(file: UploadFile = File(None), image_url: Optional[str] = Form(None)):
#     image_bytes = None
    
#     if file:
#         image_bytes = await file.read()  # Read the uploaded file
#     elif image_url:
#         image_bytes = fetch_image_from_url(image_url)  # Fetch image from URL
    
#     if image_bytes is None:
#         return {"error": "No valid image provided."}

#     # Call the ollama model with the image
#     res = ollama.chat(
#         model="llava:latest",
#         messages=[
#             {
#                 'role': 'user',
#                 'content': """
#                     You are an expert nutritionist where you need to analyze the food items
#                     and calculate the total calories. Also, provide the details of each food item with its calorie intake in the following format:
#                     1. Item 1 - no. of calories
#                     2. Item 2 - no. of calories
#                     ----
#                     ----
#                     Afterward, mention if the meal is healthy or not and include the percentage ratio of
#                     carbohydrates, proteins, fats, sugar, and calories in the meal.
#                     Finally, give suggestions on which item should be removed or added to make the
#                     meal healthier if it's unhealthy.
#                 """,
#                 'images': [image_bytes]
#             }
#         ]
#     )
#     print("-"*100)
#     print({"result": res['message']['content']})
#     print("-"*100)
#     return {"result": res['message']['content']}

# Run the application
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)



# import streamlit as st
# import base64
# import os
# from dotenv import load_dotenv
# from openai import OpenAI
# import tempfile

# load_dotenv()

# os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# client = OpenAI()

# # medical image analyzer

# sample_prompt = """You are a medical practictioner and an expert in analzying medical related images working for a very reputed hospital. You will be provided with images and you need to identify the anomalies, any disease or health issues. You need to generate the result in detailed manner. Write all the findings, next steps, recommendation, etc. You only need to respond if the image is related to a human body and health issues. You must have to answer but also write a disclaimer saying that "Consult with a Doctor before making any decisions".

# Remember, if certain aspects are not clear from the image, it's okay to state 'Unable to determine based on the provided image.'

# Now analyze the image and answer the above questions in the same structured manner defined above."""

# # calorie tracker

# sample_prompt = """
#                     You are an expert in nutritionist where you need to see the food items from the image
#                     and calculate the total calories, also provide the details of every food items with calories intake
#                     is below format
#                     1. Item 1 - no of calories
#                     2. Item 2 - no of calories
#                     ----
#                     ----
#                     After that mention that the meal is a healthy meal or not and also mention the percentage split of the ratio of
#                     carbohydrates, proteins, fats, sugar, and calories in the meal.
#                     Finally, give suggestions for which items should be removed and which items should be added to make the
#                     meal healthy if it's unhealthy.
#             """


# # Initialize session state variables
# if 'uploaded_file' not in st.session_state:
#     st.session_state.uploaded_file = None
# if 'result' not in st.session_state:
#     st.session_state.result = None

# def encode_image(image_path):
#     with open(image_path, "rb") as image_file:
#         return base64.b64encode(image_file.read()).decode('utf-8')

# def call_gpt4_model_for_analysis(filename: str, sample_prompt=sample_prompt):
#     base64_image = encode_image(filename)
    
#     messages = [
#         {
#             "role": "user",
#             "content":[
#                 {
#                     "type": "text", "text": sample_prompt
#                     },
#                 {
#                     "type": "image_url",
#                     "image_url": {
#                         "url": f"data:image/jpeg;base64,{base64_image}",
#                         "detail": "high"
#                         }
#                     }
#                 ]
#             }
#         ]

#     response = client.chat.completions.create(
#         model = "gpt-4o",
#         messages = messages,
#         max_tokens = 1500
#         )

#     print(response.choices[0].message.content)
#     return response.choices[0].message.content

# def chat_eli(query):
#     eli5_prompt = "You have to explain the below piece of information to a five years old. \n" + query
#     messages = [
#         {
#             "role": "user",
#             "content": eli5_prompt
#         }
#     ]

#     response = client.chat.completions.create(
#         model="gpt-3.5-turbo",
#         messages=messages,
#         max_tokens=1500
#     )

#     return response.choices[0].message.content

# st.title("Medical Help using Multimodal LLM")

# with st.expander("About this App"):
#     st.write("Upload an image to get an analysis from GPT-4.")

# uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

# # Temporary file handling
# if uploaded_file is not None:
#     with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
#         tmp_file.write(uploaded_file.getvalue())
#         st.session_state['filename'] = tmp_file.name

#     st.image(uploaded_file, caption='Uploaded Image')

# # Process button
# if st.button('Analyze Image'):
#     if 'filename' in st.session_state and os.path.exists(st.session_state['filename']):
#         st.session_state['result'] = call_gpt4_model_for_analysis(st.session_state['filename'])
#         st.markdown(st.session_state['result'], unsafe_allow_html=True)
#         os.unlink(st.session_state['filename'])  # Delete the temp file after processing

# # ELI5 Explanation
# if 'result' in st.session_state and st.session_state['result']:
#     st.info("Below you have an option for ELI5 to understand in simpler terms.")
#     if st.radio("ELI5 - Explain Like I'm 5", ('No', 'Yes')) == 'Yes':
#         simplified_explanation = chat_eli(st.session_state['result'])
#         st.markdown(simplified_explanation, unsafe_allow_html=True)
             



# from fastapi import FastAPI, File, UploadFile, Form
# from typing import Optional
# import requests
# import os
# import base64
# from dotenv import load_dotenv
# from openai import OpenAI
# from PIL import Image
# from fastapi import FastAPI, HTTPException
# from io import BytesIO

# app = FastAPI()

# # Load OpenAI API Key
# load_dotenv()
# os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
# client = OpenAI()


# def process_uploaded_image(uploaded_file: UploadFile):
#     if uploaded_file.content_type.startswith('image'):
#         img = Image.open(uploaded_file.file)
#         return img
#     else:
#         raise HTTPException(status_code=415, detail="Unsupported file format. Only images are allowed.")


# def process_image_url(image_url: str):
#     response = requests.get(image_url)
#     if response.status_code == 200:
#         img = Image.open(BytesIO(response.content))
#         return img
#     else:
#         raise HTTPException(status_code=404, detail="Failed to fetch the image from the provided URL.")

# # Helper function to fetch image from URL
# def fetch_image_from_url(image_url: str):
#     response = requests.get(image_url)
#     if response.status_code == 200:
#         return response.content
#     return None

# # Helper function to encode image to base64
# def encode_image(image_bytes):
#     return base64.b64encode(image_bytes).decode('utf-8')

# # Medical Image Analysis prompt
# medical_prompt = """You are a medical practictioner and an expert in analzying medical related images working for a very reputed hospital. You will be provided with images and you need to identify the anomalies, any disease or health issues. You need to generate the result in detailed manner. Write all the findings, next steps, recommendation, etc. You only need to respond if the image is related to a human body and health issues. You must have to answer but also write a disclaimer saying that "Consult with a Doctor before making any decisions".

# Remember, if certain aspects are not clear from the image, it's okay to state 'Unable to determine based on the provided image.'

# Now analyze the image and answer the above questions in the same structured manner defined above."""

# # Calorie Tracking prompt
# calorie_prompt = """
#                     You are an expert in nutritionist where you need to see the food items from the image
#                     and calculate the total calories, also provide the details of every food items with calories intake
#                     is below format
#                     1. Item 1 - no of calories
#                     2. Item 2 - no of calories
#                     ----
#                     ----
#                     After that mention that the meal is a healthy meal or not and also mention the percentage split of the ratio of
#                     carbohydrates, proteins, fats, sugar, and calories in the meal.
#                     Finally, give suggestions for which items should be removed and which items should be added to make the
#                     meal healthy if it's unhealthy.
#             """

# # Endpoint for Medical Image Analysis
# @app.post("/analyze-medical-image/")
# async def analyze_medical_image(file: UploadFile= File(None), image_url: Optional[str] = Form(None)):
#     image_bytes = None
    
#     # Check if a file was uploaded
#     if file:
#         image_bytes = process_uploaded_image(file)
#     # If no file, check if URL is provided
#     elif image_url:
#         image_bytes = fetch_image_from_url(image_url)
    
#     if image_bytes is None:
#         return {"error": "No valid image provided."}

#     # Encode image to base64
#     base64_image = encode_image(image_bytes)
    
#     # Call the model
#     messages = [
#         {
#             "role": "user",
#             "content": medical_prompt,
#             "images": [
#                 {
#                     "type": "image_url",
#                     "image_url": {
#                         "url": f"data:image/jpeg;base64,{base64_image}",
#                         "detail": "high"
#                     }
#                 }
#             ]
#         }
#     ]
    
#     response = client.chat.completions.create(
#         model="gpt-4o-mini",
#         messages=messages,
#         max_tokens=1500
#     )

#     return {"result": response.choices[0].message.content}

# # Endpoint for Calorie Tracker
# @app.post("/analyze-food-image/")
# async def analyze_food_image(file: UploadFile = File(None), image_url: Optional[str] = Form(None)):
#     image_bytes = None
    
#     # Check if a file was uploaded
#     if file:
#         image_bytes = process_uploaded_image(file)
#     # If no file, check if URL is provided
#     elif image_url:
#         image_bytes = fetch_image_from_url(image_url)
    
#     if image_bytes is None:
#         return {"error": "No valid image provided."}

#     # Encode image to base64
#     base64_image = encode_image(image_bytes)
    
#     # Call the model
#     messages = [
#         {
#             "role": "user",
#             "content": calorie_prompt,
#             "images": [
#                 {
#                     "type": "image_url",
#                     "image_url": {
#                         "url": f"data:image/jpeg;base64,{base64_image}",
#                         "detail": "high"
#                     }
#                 }
#             ]
#         }
#     ]
    
#     response = client.chat.completions.create(
#         model="gpt-4o-mini",
#         messages=messages,
#         max_tokens=1500
#     )

#     return {"result": response.choices[0].message.content}


# from fastapi import FastAPI, File, UploadFile, Form
# from typing import Optional
# import requests
# import os
# import base64
# from dotenv import load_dotenv
# from openai import OpenAI
# from PIL import Image
# from fastapi import FastAPI, HTTPException
# from io import BytesIO

# app = FastAPI()

# # Load OpenAI API Key
# load_dotenv()
# os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
# client = OpenAI()


# def process_uploaded_image(uploaded_file: UploadFile):
#     if uploaded_file.content_type.startswith('image'):
#         img = Image.open(uploaded_file.file)
#         return img
#     else:
#         raise HTTPException(status_code=415, detail="Unsupported file format. Only images are allowed.")


# def process_image_url(image_url: str):
#     response = requests.get(image_url)
#     if response.status_code == 200:
#         img = Image.open(BytesIO(response.content))
#         return img
#     else:
#         raise HTTPException(status_code=404, detail="Failed to fetch the image from the provided URL.")

# # Helper function to fetch image from URL
# def fetch_image_from_url(image_url: str):
#     response = requests.get(image_url)
#     if response.status_code == 200:
#         return response.content
#     return None

# # Helper function to encode image to base64
# def encode_image(image: Image.Image):
#     buffered = BytesIO()
#     image.save(buffered, format="JPEG")
#     return base64.b64encode(buffered.getvalue()).decode('utf-8')

# # Medical Image Analysis prompt
# medical_prompt = """You are a medical practictioner and an expert in analyzing medical-related images working for a very reputed hospital. You will be provided with images and you need to identify the anomalies, any disease or health issues. You need to generate the result in detailed manner. Write all the findings, next steps, recommendations, etc. You only need to respond if the image is related to a human body and health issues. You must have to answer but also write a disclaimer saying that "Consult with a Doctor before making any decisions".

# Remember, if certain aspects are not clear from the image, it's okay to state 'Unable to determine based on the provided image.'

# Now analyze the image and answer the above questions in the same structured manner defined above."""

# # Calorie Tracking prompt
# calorie_prompt = """
#                     You are an expert nutritionist where you need to see the food items from the image
#                     and calculate the total calories. Also, provide the details of every food item with calories intake
#                     in the below format:
#                     1. Item 1 - no of calories
#                     2. Item 2 - no of calories
#                     ----
#                     ----
#                     After that, mention if the meal is a healthy meal or not and also mention the percentage split of the ratio of
#                     carbohydrates, proteins, fats, sugar, and calories in the meal.
#                     Finally, give suggestions for which items should be removed and which items should be added to make the
#                     meal healthy if it's unhealthy.
#             """

# # Endpoint for Medical Image Analysis
# @app.post("/analyze-medical-image/")
# async def analyze_medical_image(file: UploadFile = File(None), image_url: Optional[str] = Form(None)):
#     image = None
    
#     # Check if a file was uploaded
#     if file:
#         image = process_uploaded_image(file)
#     # If no file, check if URL is provided
#     elif image_url:
#         image = process_image_url(image_url)
    
#     if image is None:
#         return {"error": "No valid image provided."}

#     # Encode image to base64
#     base64_image = encode_image(image)
#     print(base64_image)
    
#     # Call the model
#     messages = [
#         {
#             "role": "user",
#             "content": medical_prompt,
#             "images": [
#                 {
#                     "type": "image_url",
#                     "image_url": {
#                         "url": f"data:image/jpeg;base64,{base64_image}",
#                         "detail": "high"
#                     }
#                 }
#             ]
#         }
#     ]
    
#     response = client.chat.completions.create(
#         model="gpt-4o",
#         messages=messages,
#         max_tokens=1500
#     )

#     return {"result": response.choices[0].message.content}

# # Endpoint for Calorie Tracker
# @app.post("/analyze-food-image/")
# async def analyze_food_image(file: UploadFile = File(None), image_url: Optional[str] = Form(None)):
#     image = None
    
#     # Check if a file was uploaded
#     if file:
#         image = process_uploaded_image(file)
#     # If no file, check if URL is provided
#     elif image_url:
#         image = process_image_url(image_url)
    
#     if image is None:
#         return {"error": "No valid image provided."}

#     # Encode image to base64
#     base64_image = encode_image(image)
    
#     # Call the model
#     messages = [
#         {
#             "role": "user",
#             "content": calorie_prompt,
#             "images": [
#                 {
#                     "type": "image_url",
#                     "image_url": {
#                         "url": f"data:image/jpeg;base64,{base64_image}",
#                         "detail": "high"
#                     }
#                 }
#             ]
#         }
#     ]
    
#     response = client.chat.completions.create(
#         model="gpt-4o",
#         messages=messages,
#         max_tokens=1500
#     )

#     return {"result": response.choices[0].message.content}


