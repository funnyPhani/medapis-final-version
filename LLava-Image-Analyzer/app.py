# import streamlit as st
# from config import Config
# from helpers.image_helper import create_temp_file
# from helpers.llm_helper import analyze_image_file, stream_parser

# page_title = Config.PAGE_TITLE

# # configures page settings
# st.set_page_config(
#     page_title=page_title,
#     initial_sidebar_state="expanded",
# )

# # page title
# st.title(page_title)

# st.markdown("#### Select an image file to analyze.")

# # displays file upload widget
# uploaded_file = st.file_uploader("Choose image file", type=['png', 'jpg', 'jpeg'] )

# # sets up sidebar nav widgets
# with st.sidebar:   
#     # creates selectbox to pick the model we would like to use
#     image_model = st.selectbox('Which image model would you like to use?', Config.OLLAMA_MODELS)

# if chat_input := st.chat_input("What would you like to ask?"):
#     if uploaded_file is None:
#         st.error('You must select an image file to analyze!')
#         st.stop()

#     # Color formatting example https://docs.streamlit.io/library/api-reference/text/st.markdown
#     with st.status(":red[Processing image file. DON'T LEAVE THIS PAGE WHILE IMAGE FILE IS BEING ANALYZED...]", expanded=True) as status:
#         st.write(":orange[Analyzing Image File...]")

#         # creates the audio file
#         stream = analyze_image_file(uploaded_file, model=image_model, user_prompt=chat_input)
       
#         stream_output = st.write_stream(stream_parser(stream))

#         st.write(":green[Done analyzing image file]")


import streamlit as st
from config import Config
from helpers.image_helper import create_temp_file
from helpers.llm_helper import analyze_image_file, stream_parser

# Page title from config
page_title = Config.PAGE_TITLE

# Configures page settings
st.set_page_config(
    page_title=page_title,
    initial_sidebar_state="expanded",
)

# Page title
st.title(page_title)
st.markdown("#### Select an image file to analyze.")

# Displays file upload widget
uploaded_file = st.file_uploader("Choose an image file", type=['png', 'jpg', 'jpeg'])
# def clear_text():
#     st.session_state.my_text = ""
# prompt = st.text_input("User prompt", on_change=clear_text)
# Sets up sidebar nav widgets
# with st.sidebar:
#     # Creates a selectbox to pick the model to use
#     image_model = st.selectbox('Which image model would you like to use?', Config.OLLAMA_MODELS)
image_model = "llava:latest"
# Logic after file upload
if uploaded_file:
    if uploaded_file is None:
        st.error('You must select an image file to analyze!')
        st.stop()

    # Let the user choose between the "Medical Image Analyzer" and "Calorie Tracker"
    st.markdown("#### Choose an analysis type:")
    col1, col2 = st.columns(2)

    # Create two buttons
    medical_button = col1.button("Medical Image Analyzer")
    calorie_button = col2.button("Calorie Tracker")

    # Logic based on which button is clicked
    if medical_button:
        # Medical image analysis prompt
        chat_input = """
            You are a medical practitioner and an expert in analyzing medical-related images working for a very reputed hospital. 
            You will be provided with images and you need to identify the anomalies, any disease or health issues. 
            You need to generate the result in a detailed manner. 
            Write all the findings, next steps, recommendation, etc. You only need to respond if the image is related to a human body and health issues.
            You must have to answer but also write a disclaimer saying that "Consult with a Doctor before making any decisions. 
        """

        with st.status(":red[Processing image file. DON'T LEAVE THIS PAGE WHILE IMAGE FILE IS BEING ANALYZED...]", expanded=True) as status:
            st.write(":orange[Analyzing Medical Image...]")
            stream = analyze_image_file(uploaded_file, model=image_model, user_prompt=chat_input)
            stream_output = st.write_stream(stream_parser(stream))
            st.write(":green[Done analyzing image file]")

    elif calorie_button:
        # Calorie tracking analysis prompt
        chat_input = """
                        You are an expert in nutritionist where you need to see the food items from the image
                        and calculate the total calories, also provide the details of every food items with calories intake
                        is below format
                        1. Item 1 with name - no of calories
                        2. Item 1 with name - no of calories
                        ----
                        ----
                        After that mention that the meal is a healthy meal or not and also mention the percentage split of the ratio of
                        carbohydrates, proteins, fats, sugar, and calories in the meal.
                        Finally, give suggestions for which items should be removed and which items should be added to make the
                        meal healthy if it's unhealthy. 
                    """

        with st.status(":red[Processing image file. DON'T LEAVE THIS PAGE WHILE IMAGE FILE IS BEING ANALYZED...]", expanded=True) as status:
            st.write(":orange[Analyzing Calorie Information...]")
            stream = analyze_image_file(uploaded_file, model=image_model, user_prompt=chat_input)
            stream_output = st.write_stream(stream_parser(stream))
            st.write(":green[Done analyzing image file]")

    else:
        st.warning("Please choose either 'Medical Image Analyzer' or 'Calorie Tracker' to proceed.")



