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





from fastapi import FastAPI, File, UploadFile, Form
from typing import Optional
import ollama
from fastapi.responses import HTMLResponse
import requests

app = FastAPI()

# Helper function to fetch image from URL
def fetch_image_from_url(image_url: str):
    response = requests.get(image_url)
    if response.status_code == 200:
        return response.content
    return None



# Endpoint 1: Analyze medical image (file or URL)
@app.post("/analyze-medical-image/")
async def analyze_medical_image(file: UploadFile = File(None), image_url: Optional[str] = Form(None)):
    image_bytes = None
    
    if file:
        image_bytes = await file.read()  # Read the uploaded file
    elif image_url:
        image_bytes = fetch_image_from_url(image_url)  # Fetch image from URL
    
    if image_bytes is None:
        return {"error": "No valid image provided."}

    # Call the ollama model with the image
    res = ollama.chat(
        model="llava:latest",
        messages=[
            {
                'role': 'user',
                'content': """
                    You are a medical practitioner and an expert in analyzing medical images. 
                    Identify any anomalies, diseases, or health issues in the image. 
                    Provide detailed findings, recommendations, and next steps. 
                    If certain aspects are unclear, state 'Unable to determine based on the provided image.' 
                    Also, include a disclaimer: 'Consult with a doctor before making any decisions.'
                    Provide the information in only markdown format.
                """,
                'images': [image_bytes]
            }
        ]
    )
    print("-"*100)
    print({"result": res['message']['content']})
    print("-"*100)

    return {"result": res['message']['content']}

# Endpoint 2: Analyze food image for nutrition (file or URL)
@app.post("/analyze-food-image/")
async def analyze_food_image(file: UploadFile = File(None), image_url: Optional[str] = Form(None)):
    image_bytes = None
    
    if file:
        image_bytes = await file.read()  # Read the uploaded file
    elif image_url:
        image_bytes = fetch_image_from_url(image_url)  # Fetch image from URL
    
    if image_bytes is None:
        return {"error": "No valid image provided."}

    # Call the ollama model with the image
    res = ollama.chat(
        model="llava:latest",
        messages=[
            {
                'role': 'user',
                'content': """
                    You are an expert nutritionist where you need to analyze the food items
                    and calculate the total calories. Also, provide the details of each food item with its calorie intake in the following format:
                    1. Item 1 - no. of calories
                    2. Item 2 - no. of calories
                    ----
                    ----
                    Afterward, mention if the meal is healthy or not and include the percentage ratio of
                    carbohydrates, proteins, fats, sugar, and calories in the meal.
                    Finally, give suggestions on which item should be removed or added to make the
                    meal healthier if it's unhealthy.
                """,
                'images': [image_bytes]
            }
        ]
    )
    print("-"*100)
    print({"result": res['message']['content']})
    print("-"*100)
    return {"result": res['message']['content']}

# Run the application
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)



