from rich import print
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image
import requests
import os
from io import BytesIO
import ollama
import chromadb
import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import ollama
from fastapi.responses import JSONResponse
from starlette.responses import RedirectResponse
import google.generativeai as genai

from fastapi import FastAPI, File, UploadFile, Form
from typing import Optional
import ollama
from fastapi.responses import HTMLResponse
import requests
from dotenv import load_dotenv
load_dotenv()
import yaml
# Configure Google Gemini AI
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Load the configuration from the YAML file
with open('config.yml', 'r') as file:
    config = yaml.safe_load(file)

# Access the model names
llm_model_name = config['llm_model_name']
embd_model_name = config['embd_model_name']
groq_model = config['groq_model']
google_model = config['google_model']
ollama_vision_model = config["ollama_vision_model"]
print(f'LLM Model Name: {llm_model_name}')
print(f'Embedding Model Name: {embd_model_name}')
print(f'groq_model Name: {groq_model}')
print(f'google_model Name: {google_model}')
print(f'ollama_vision_model Name: {ollama_vision_model}')




def process_uploaded_image(uploaded_file: UploadFile):
    if uploaded_file.content_type.startswith('image'):
        img = Image.open(uploaded_file.file)
        return img
    else:
        raise HTTPException(status_code=415, detail="Unsupported file format. Only images are allowed.")


def process_image_url(image_url: str):
    response = requests.get(image_url)
    if response.status_code == 200:
        img = Image.open(BytesIO(response.content))
        return img
    else:
        raise HTTPException(status_code=404, detail="Failed to fetch the image from the provided URL.")
    

def get_gemini_response(input_text: str, image_data: Image.Image):
    model = genai.GenerativeModel(google_model)
    response = model.generate_content([input_text, image_data])
    return response.text

app = FastAPI()

# Initialize ChromaDB client for RAG system
client = chromadb.PersistentClient(path="db/")
collection = client.get_collection("readme_collection")


def retrieve_document(query: str, n_results: int):
    query_embedding = ollama.embeddings(
        model=embd_model_name,
        prompt=query,
    )
    results = collection.query(
        query_embeddings=[query_embedding['embedding']],
        n_results=n_results
    )
    if results['documents']:
        return results['documents'][0], results['ids'][0]
    else:
        return "No relevant document found", None

def rag_system_groq1(query: str, n_results: int):
   try:
      document_content, document_id = retrieve_document(query, n_results)
      finalData = []

      if document_id is None:
         return "No relevant document found"
      import ollama

      for res, doc_name in zip(document_content, document_id):
        try:
            res = ollama.chat(
            model=llm_model_name,
            messages=[
               {
                  'role': 'user',
                  'content': f"""You are an expert in analyzing queries and their relevant context. Given the query: {query} and the context: {res}, generate a concise and accurate response. 
                  If the provided context does not offer meaningful information for the query, create an appropriate, concise short response in the markdown format."""
               }
            ]
      )
            finalData.append({
                "result": res['message']['content'],
                "document_id": doc_name
            })
        except Exception as e:
            return f"An error occurred: {str(e)}"
   except Exception as e:
        print(e)

   print("-"*100)
   print("Query:",query)
   print("-"*100)
   print("Response:",finalData)
   print("-"*100)
   return finalData


@app.get("/", tags=["Test the MedApp-Apis with Open-source"])
async def index():
    return RedirectResponse(url="/docs")

# Helper function to fetch image from URL
def fetch_image_from_url(image_url: str):
    response = requests.get(image_url)
    if response.status_code == 200:
        return response.content
    return None



# Endpoint 1: Analyze medical image (file or URL)
@app.post("/analyze-medical-image", tags=["Medical Image Analyzer"])
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
        model= ollama_vision_model,
        messages=[
            {
                'role': 'user',
                'content': """
                    You are a medical practitioner and an expert in analyzing medical images. 
                    Identify any anomalies, diseases, or health issues in the image. 
                    Provide detailed findings, recommendations, and next steps. 
                    If certain aspects are unclear, state 'Unable to determine based on the provided image.' 
                    Also, include a disclaimer: 'Consult with a doctor before making any decisions.'
                    Provide the response  only in markdown format.
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
@app.post("/analyze-food-image", tags=["Meal Analyzer"])
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
        model=ollama_vision_model,
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
                    Provide the response  only in markdown format.
                """,
                'images': [image_bytes]
            }
        ]
    )
    print("-"*100)
    print({"result": res['message']['content']})
    print("-"*100)
    return {"result": res['message']['content']}

class QueryRequest(BaseModel):
    query: str
    n_results: int

@app.post("/nhs_rag", tags=["Test the NHS data using RAG"])
async def get_response(request: QueryRequest):
    try:
        query = request.query
        n_results = request.n_results
        response = rag_system_groq1(query, n_results)
        print("-" * 100)
        print("Query:", request.query)
        print("-" * 100)
        print("Response:", response)
        print("-" * 100)
        return response
    except Exception as e:
        return {"error": str(e)}
    

class QueryRequest(BaseModel):
    query: str

@app.post("/medqa",tags=["Medical Q&A"])
async def generate_response(request: QueryRequest):
    try:
        response = ollama.chat(
            model=llm_model_name,
            messages=[
                {
                    'role': 'user',
                    'content': f"You are a medical advisor. Try to respond to the user query in short: {request.query}. Try to generate the response in Markdown format only."

                }
            ]
        )
        response1 = ollama.chat(
            model=llm_model_name,
            messages=[
                {
                    'role': 'user',
                    'content': f"Based on the user query: {request.query}, please generate two connected questions in a list format and do not provide a preamble."

                }
            ]
        )
        # updated code
        CQ = {"connected_questions": response1['message']['content']}
        result = {"response": response['message']['content']}
        result.update(CQ)
        print("-"*100)
        print("Query:",request.query)
        print("-"*100)
        print("Response :", result)
        print("-"*100)
        

        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")
    
   #      print("-"*100)
   #      print("Query:",request.query)
   #      print("-"*100)
   #      print("Response :", response['message']['content'])
   #      print("-"*100)
        

   #      return JSONResponse(content={"response": response['message']['content']})
   #  except Exception as e:
   #      raise HTTPException(status_code=500, detail=f"An error occurred: {e}")
    


class QueryRequest(BaseModel):
    query: str

@app.post("/qa",tags=["General Q&A"])
async def generate_response(request: QueryRequest):
    try:
        response = ollama.chat(
            model=llm_model_name,
            messages=[
                {
                    'role': 'user',
                    'content': f"Please respond to the user query: {request.query} in a very concise Markdown format only."

                }
            ]
        )
        response1 = ollama.chat(
            model=llm_model_name,
            messages=[
                {
                    'role': 'user',
                    'content': f"Based on the user query: {request.query}, please generate two connected questions in a list format and do not provide a preamble."

                }
            ]
        )
        CQ = {"connected_questions": response1['message']['content']}
        result = {"response": response['message']['content']}
        result.update(CQ)
        print("-"*100)
        print("Query:",request.query)
        print("-"*100)
        print("Response :", result)
        print("-"*100)
        

        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")




class QueryRequests(BaseModel):
    ds: str

@app.post("/dsg",tags=["Discharge summary generation"])
async def generate_ds(request: QueryRequests):
    try:
        response = ollama.chat(
            model=llm_model_name,
            messages=[
                {
                    'role': 'user',
                    'content': f"""
As a medical advisor, please review the provided discharge summary {request.ds} and extract the following key details. Present the information in the structured format below:

1. **Patient Information**
   - **Age**: [Insert patient's age]
   - **Gender**: [Insert patient's gender]
   - **Patient ID**: [Insert patient ID]

2. **Diagnosis**
   - **Provisional or Final**: [Insert diagnosis]

3. **Patient Complaints**
   - Insert symptoms
   - Conditions, or concerns that led to the hospital admission

4. **Past History or Past Medical History**
   - **Chronic Conditions**: [Insert chronic conditions]
   - **Previous Surgeries**: [Insert details of previous surgeries]
   - **Other Relevant Health Information**: [Insert any other relevant health information]

5. **Personal & Family History**
   - **Personal History**: [Insert personal health history]
   - **Family History**: [Insert family health history]

   a. **Social History**
      - **Lifestyle Factors**: [Insert details about lifestyle factors]
      - **Substance Use**: [Insert information about substance use]

6. **Examination Findings (Clinical findings)**
   - **Physical Examination**: [Insert details from the physical examination]

      i. **Vitals**: [Insert vital signs]
      ii. **CVS (Cardiovascular System)**: [Insert cardiovascular system findings]
      - **RS (Respiratory System)**: [Insert respiratory system findings]
      - **P/A (Per Abdomen)**: [Insert abdominal examination findings]
      - **CNS (Central Nervous System)**: [Insert central nervous system findings]
      iii. **Local Examination**: [Insert details from any local examination, if available]

7. **Investigations**
   - **Diagnostic Tests**: [Insert results of diagnostic tests]

8. **Course in the Hospital**
   - **Treatment and Progress**: [Insert details on treatment and patient progress during the hospital stay]
      a. **Medication Given**: [Insert list of medications administered]
      b. **Procedure Notes & Details (For Surgery Only)**: [Insert details of any surgical procedures performed]
      c. **Post-operative Treatment (For Surgery Only)**: [Insert details on post-operative care]

9. **Discharge Advice**
   - **General Advice**: [Insert general advice provided at discharge]
      a. **Medication**: [Insert medication instructions at discharge]
      b. **Follow-up Details**: [Insert follow-up appointments or instructions]
         i. **Ward Number**: [Insert ward number]
         ii. **Ambulance Number**: [Insert ambulance number, if applicable]

Please ensure that all extracted information is presented clearly and accurately according to the details in the discharge summary.
                    """
                }
            ]
        )
        print("-" * 100)
        print("Discharge Summary:", request.ds)
        print("-" * 100)
        print("Response:", response['message']['content'])
        print("-" * 100)
        
        return JSONResponse(content={"response": response['message']['content']})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")



@app.get("/close", tags=["Test the MedApp-Apis with Close-source"])
async def index():
    return "Using close-source model to get the responses."


def rag_system_groq(query: str, n_results: int):
    import os
    from groq import Groq
    api = os.getenv("GROQ_API_KEY")
    client = Groq(api_key=api)

    document_content, document_id = retrieve_document(query, n_results)
    finalData = []

    if document_id is None:
        return "No relevant document found"

    for res, doc_name in zip(document_content, document_id):
        try:
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": f"You are an expert in analyzing queries and their relevant context. Given the query: {query} and the context: {res}, generate a concise and accurate response. If the provided context {res} does not offer meaningful information for the query {query}, create an appropriate, concise response."
                    }
                ],
                model=groq_model
            )
            finalData.append({
                "result": chat_completion.choices[0].message.content,
                "document_id": doc_name
            })
        except Exception as e:
            return f"An error occurred: {str(e)}"

    return finalData


class QueryRequest(BaseModel):
    query: str
    n_results: int

@app.post("/rag-response",tags=["Test the NHS data using RAG Approach"])
async def get_response(request: QueryRequest):
    try:
        query = request.query
        n_results = request.n_results
        response = rag_system_groq(query, n_results)
        print("-" * 100)
        print("Query:", request.query)
        print("-" * 100)
        print("Response:", response)
        print("-" * 100)
        return response
    except Exception as e:
        return {"error": str(e)}
    
class QueryRequest(BaseModel):
    query: str

@app.post("/med_qa",tags=["Medical QA "])
async def generate_response(request: QueryRequest):
    try:
        from groq import Groq
        api = os.getenv("GROQ_API_KEY")
        client = Groq(api_key=api)
        response = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": f"You are a medical advisor. Try to respond to the user query in short: {request.query}. Try to generate the response in Markdown format only."
                    }
                ],
                model=groq_model
            )
        response1 = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": f"Based on the user query: {request.query}, please generate two connected questions in a list format and do not provide a preamble."
                    }
                ],
                model=groq_model
            )
        
        
        # updated code
        CQ = {"connected_questions": response1.choices[0].message.content}
        result = {"response": response.choices[0].message.content}
        result.update(CQ)
        print("-"*100)
        print("Query:",request.query)
        print("-"*100)
        print("Response :", result)
        print("-"*100)
        

        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")
    
class QueryRequest(BaseModel):
    query: str
    n_results: int

# @app.post("/rag-response",tags=["Test the NHS data using RAG Approach"])
# async def get_response(request: QueryRequest):
#     try:
#         query = request.query
#         n_results = request.n_results
#         response = rag_system_groq(query, n_results)
#         print("-" * 100)
#         print("Query:", request.query)
#         print("-" * 100)
#         print("Response:", response)
#         print("-" * 100)
#         return response
#     except Exception as e:
#         return {"error": str(e)}
    
class QueryRequest(BaseModel):
    query: str

@app.post("/gen_qa",tags=["General QA "])
async def generate_response(request: QueryRequest):
    try:
        from groq import Groq
        api = os.getenv("GROQ_API_KEY")
        client = Groq(api_key=api)
        response = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": f"Please respond to the user query: {request.query} in a very concise Markdown format only."
                    }
                ],
                model=groq_model
            )
        response1 = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": f"Based on the user query: {request.query}, please generate two connected questions in a list format and do not provide a preamble."
                    }
                ],
                model=groq_model
            )
        
        
        # updated code
        CQ = {"connected_questions": response1.choices[0].message.content}
        result = {"response": response.choices[0].message.content}
        result.update(CQ)
        print("-"*100)
        print("Query:",request.query)
        print("-"*100)
        print("Response :", result)
        print("-"*100)
        

        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")
    

class QueryRequests(BaseModel):
    ds: str

@app.post("/gds",tags=["Discharge summary generator"])
async def generate_ds(request: QueryRequests):
    try:
        from groq import Groq
        api = os.getenv("GROQ_API_KEY")
        client = Groq(api_key=api)
        response = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": f"""
As a medical advisor, please review the provided discharge summary {request.ds} and extract the following key details. Present the information in the structured format below without preamble:

1. **Patient Information**
   - **Age**: [Insert patient's age]
   - **Gender**: [Insert patient's gender]
   - **Patient ID**: [Insert patient ID]

2. **Diagnosis**
   - **Provisional or Final**: [Insert diagnosis]

3. **Patient Complaints**
   - Insert symptoms
   - Conditions, or concerns that led to the hospital admission

4. **Past History or Past Medical History**
   - **Chronic Conditions**: [Insert chronic conditions]
   - **Previous Surgeries**: [Insert details of previous surgeries]
   - **Other Relevant Health Information**: [Insert any other relevant health information]

5. **Personal & Family History**
   - **Personal History**: [Insert personal health history]
   - **Family History**: [Insert family health history]

   a. **Social History**
      - **Lifestyle Factors**: [Insert details about lifestyle factors]
      - **Substance Use**: [Insert information about substance use]

6. **Examination Findings (Clinical findings)**
   - **Physical Examination**: [Insert details from the physical examination]

      i. **Vitals**: [Insert vital signs]
      ii. **CVS (Cardiovascular System)**: [Insert cardiovascular system findings]
      - **RS (Respiratory System)**: [Insert respiratory system findings]
      - **P/A (Per Abdomen)**: [Insert abdominal examination findings]
      - **CNS (Central Nervous System)**: [Insert central nervous system findings]
      iii. **Local Examination**: [Insert details from any local examination, if available]

7. **Investigations**
   - **Diagnostic Tests**: [Insert results of diagnostic tests]

8. **Course in the Hospital**
   - **Treatment and Progress**: [Insert details on treatment and patient progress during the hospital stay]
      a. **Medication Given**: [Insert list of medications administered]
      b. **Procedure Notes & Details (For Surgery Only)**: [Insert details of any surgical procedures performed]
      c. **Post-operative Treatment (For Surgery Only)**: [Insert details on post-operative care]

9. **Discharge Advice**
   - **General Advice**: [Insert general advice provided at discharge]
      a. **Medication**: [Insert medication instructions at discharge]
      b. **Follow-up Details**: [Insert follow-up appointments or instructions]
         i. **Ward Number**: [Insert ward number]
         ii. **Ambulance Number**: [Insert ambulance number, if applicable]

Please ensure that all extracted information is presented clearly and accurately according to the details in the discharge summary.
                    """
                    }
                ],
                model=groq_model
            )
        
               
        print("-" * 100)
        print("Discharge Summary:", request.ds)
        print("-" * 100)
        print("Response:", response.choices[0].message.content)
        print("-" * 100)
        
        return JSONResponse(content={"response": response.choices[0].message.content})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")
    



@app.post("/analyze-med-image", tags=["Analyze the medical image"])
async def analyze_image(
    file: UploadFile = File(None),
    image_url: str = Form(None),
    prompt: str = Form(""),
    use_eli5_prompt: bool = Form(False)
):
    if image_url:
        image_data = process_image_url(image_url)
    elif file:
        image_data = process_uploaded_image(file)
    else:
        raise HTTPException(status_code=400, detail="Neither image nor image URL provided.")

    if use_eli5_prompt:
        prompt = "You have to explain the below piece of information to a five-year-old. \n" + prompt
    else:
        prompt = f"""
        You are a medical practitioner and an expert in analyzing medical images. Identify any anomalies, diseases, or health issues in the image. Provide detailed findings, recommendations, and next steps. If certain aspects are unclear, state 'Unable to determine based on the provided image.' Also, include a disclaimer: 'Consult with a doctor before making any decisions.'
        {prompt}
        """

    response_text = get_gemini_response(prompt, image_data)

    if use_eli5_prompt:
        eli5_prompt = "You have to explain the below piece of information to a five-year-old. \n" + response_text
        response_text = get_gemini_response(eli5_prompt, image_data)

    return JSONResponse(content={"response": response_text})


@app.post("/analyze-meal", tags=["Analyze the meal"])
async def analyze_meal(image: UploadFile = File(None), image_url: str = Form(None)):
    if image:
        image_data = process_uploaded_image(image)
    elif image_url:
        image_data = process_image_url(image_url)
    else:
        raise HTTPException(status_code=400, detail="Neither image nor image URL provided.")

    # input_prompt = """
    # You are a nutrition expert. Analyze the food items in the image, calculate the total calories, and provide a detailed breakdown of the calories for each item. Indicate whether the meal is healthy or not, including the percentage split of carbohydrates, proteins, fats, sugars, and calories. Offer suggestions for removing or adding items to make the meal healthier if necessary.
    # """
    input_prompt = """
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


    response_text = get_gemini_response(input_prompt, image_data)
    return {"response": response_text}



###################
def rag_system_groq3(query: str, n_results: int):
    document_content, document_id = retrieve_document(query, n_results)
    finalData = []
    import google.generativeai as genai
    import os
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    model = genai.GenerativeModel(model_name=google_model)
    
    if document_id is None:
        return "No relevant document found"

    for res, doc_name in zip(document_content, document_id):
        try:
            response = model.generate_content(f"You are an expert in analyzing queries and their relevant context. Given the query: {query} and the context: {res}, generate a concise and accurate response. If the provided context {res} does not offer meaningful information for the query {query}, create an appropriate, concise response.")
                
            finalData.append({
                "result": response.text,
                "document_id": doc_name
            })
        except Exception as e:
            return f"An error occurred: {str(e)}"

    return finalData

class QueryRequest(BaseModel):
    query: str
    n_results: int

@app.post("/rag-response-gemini",tags=["Test the NHS data using RAG Approach Gemini"])
async def get_response(request: QueryRequest):
    try:
        query = request.query
        n_results = request.n_results
        response = rag_system_groq3(query, n_results)
        print("-" * 100)
        print("Query:", request.query)
        print("-" * 100)
        print("Response:", response)
        print("-" * 100)
        return response
    except Exception as e:
        return {"error": str(e)}
    


class QueryRequest(BaseModel):
    query: str

@app.post("/med_qa_gemini",tags=["Medical QA using Gemini"])
async def generate_response(request: QueryRequest):
    try:
        import google.generativeai as genai
        import os
        genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
        model = genai.GenerativeModel(model_name=google_model)
        response = model.generate_content(f"You are a medical advisor. Try to respond to the user query in short: {request.query}. Try to generate the response in Markdown format only.")
        response1 = model.generate_content(f"Based on the user query: {request.query}, please generate two connected questions in a list format and do not provide a preamble.")
        
        
        # updated code
        CQ = {"connected_questions": response1.text}
        result = {"response": response.text}
        result.update(CQ)
        print("-"*100)
        print("Query:",request.query)
        print("-"*100)
        print("Response :", result)
        print("-"*100)
        

        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")
    

class QueryRequest(BaseModel):
    query: str

@app.post("/gen_qa_gemini",tags=["Medical QA Geminis"])
async def generate_response(request: QueryRequest):
    try:
        import google.generativeai as genai
        import os
        genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
        model = genai.GenerativeModel(model_name=google_model)
        response = model.generate_content(f"Please respond to the user query: {request.query} in a very concise Markdown format only.")
        response1 = model.generate_content(f"Based on the user query: {request.query}, please generate two connected questions in a list format and do not provide a preamble.")
        
        
        # updated code
        CQ = {"connected_questions": response1.text}
        result = {"response": response.text}
        result.update(CQ)
        print("-"*100)
        print("Query:",request.query)
        print("-"*100)
        print("Response :", result)
        print("-"*100)
        

        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")
    
class QueryRequests(BaseModel):
    ds: str

@app.post("/gds-gemini",tags=["Discharge summary generator using Gemini"])
async def generate_ds(request: QueryRequests):
    try:
        import google.generativeai as genai
        import os
        genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
        model = genai.GenerativeModel(model_name=google_model)
        response = model.generate_content(f"""
As a medical advisor, please review the provided discharge summary {request.ds} and extract the following key details. Present the information in the structured format below without preamble:

1. **Patient Information**
   - **Age**: [Insert patient's age]
   - **Gender**: [Insert patient's gender]
   - **Patient ID**: [Insert patient ID]

2. **Diagnosis**
   - **Provisional or Final**: [Insert diagnosis]

3. **Patient Complaints**
   - Insert symptoms
   - Conditions, or concerns that led to the hospital admission

4. **Past History or Past Medical History**
   - **Chronic Conditions**: [Insert chronic conditions]
   - **Previous Surgeries**: [Insert details of previous surgeries]
   - **Other Relevant Health Information**: [Insert any other relevant health information]

5. **Personal & Family History**
   - **Personal History**: [Insert personal health history]
   - **Family History**: [Insert family health history]

   a. **Social History**
      - **Lifestyle Factors**: [Insert details about lifestyle factors]
      - **Substance Use**: [Insert information about substance use]

6. **Examination Findings (Clinical findings)**
   - **Physical Examination**: [Insert details from the physical examination]

      i. **Vitals**: [Insert vital signs]
      ii. **CVS (Cardiovascular System)**: [Insert cardiovascular system findings]
      - **RS (Respiratory System)**: [Insert respiratory system findings]
      - **P/A (Per Abdomen)**: [Insert abdominal examination findings]
      - **CNS (Central Nervous System)**: [Insert central nervous system findings]
      iii. **Local Examination**: [Insert details from any local examination, if available]

7. **Investigations**
   - **Diagnostic Tests**: [Insert results of diagnostic tests]

8. **Course in the Hospital**
   - **Treatment and Progress**: [Insert details on treatment and patient progress during the hospital stay]
      a. **Medication Given**: [Insert list of medications administered]
      b. **Procedure Notes & Details (For Surgery Only)**: [Insert details of any surgical procedures performed]
      c. **Post-operative Treatment (For Surgery Only)**: [Insert details on post-operative care]

9. **Discharge Advice**
   - **General Advice**: [Insert general advice provided at discharge]
      a. **Medication**: [Insert medication instructions at discharge]
      b. **Follow-up Details**: [Insert follow-up appointments or instructions]
         i. **Ward Number**: [Insert ward number]
         ii. **Ambulance Number**: [Insert ambulance number, if applicable]

Please ensure that all extracted information is presented clearly and accurately according to the details in the discharge summary.
                    """)
        
               
        print("-" * 100)
        print("Discharge Summary:", request.ds)
        print("-" * 100)
        print("Response:", response.text)
        print("-" * 100)
        
        return JSONResponse(content={"response": response.text})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")
    