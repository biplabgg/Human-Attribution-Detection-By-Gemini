# import streamlit as st
# import google.generativeai as genai
# import os
# import PIL.Image

# # Set API Key for Google Gemini
# os.environ["GOOGLE_API_KEY"] = os.getenv('GOOGLE_API_KEY')
# genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# # Load the Gemini Model
# model = genai.GenerativeModel("models/gemini-1.5-flash-latest")


# # Function to analyze human attributes
# def analyze_human_attributes(image):
#     prompt = """
#     You are an AI trained to analyze human attributes from images with high accuracy. 
#     Carefully analyze the given image and return the following structured details:

#     You have to return all results as you have the image, don't want any apologize or empty results.

#     - **Gender** (Male/Female/Non-binary)
#     - **Age Estimate** (e.g., 25 years)
#     - **Ethnicity** (e.g., Asian, Caucasian, African, etc.)
#     - **Mood** (e.g., Happy, Sad, Neutral, Excited)
#     - **Facial Expression** (e.g., Smiling, Frowning, Neutral, etc.)
#     - **Glasses** (Yes/No)
#     - **Beard** (Yes/No)
#     - **Hair Color** (e.g., Black, Blonde, Brown)
#     - **Eye Color** (e.g., Blue, Green, Brown)
#     - **Headwear** (Yes/No, specify type if applicable)
#     - **Emotions Detected** (e.g., Joyful, Focused, Angry, etc.)
#     - **Confidence Level** (Accuracy of prediction in percentage)
#     """
#     response = model.generate_content([prompt, image])
#     return response.text.strip()


# # Streamlit App
# st.title("Human Attribute Detection")
# st.write("Upload an image to detect human attributes with AI.")

# # Image Upload
# uploaded_image = st.file_uploader("Upload an Image", type=['png', 'jpg', 'jpeg'])

# if uploaded_image:
#     img = PIL.Image.open(uploaded_image)
#     person_info = analyze_human_attributes(img)

#     # Create two columns for side-by-side display
#     col1, col2 = st.columns(2)

#     with col1:
#         st.image(img, caption="Uploaded Image", use_container_width=True)

#     with col2:
#         st.write(person_info)

import streamlit as st
import google.generativeai as genai
import os
import PIL.Image
import re
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# Load Gemini Model
model = genai.GenerativeModel("models/gemini-1.5-flash-latest")

# Streamlit Page Configuration
st.set_page_config(page_title="Be An AI", layout="wide")

# Sidebar Instructions
with st.sidebar:
    st.header("🔍 Instructions")
    st.markdown("""
    1. Upload an image of a person.
    2. Click the **Submit** button to analyze attributes.
    3. Results will appear on the right.
    """)
    st.markdown("**🔹 Powered by Google Gemini AI**")
    st.markdown("**🔹 Devloped By Biplab**")

# Function to analyze human attributes
def analyze_human_attributes(image):
    try:
        prompt = """
        You are an AI trained to analyze human attributes from images with high accuracy. 
        Carefully analyze the given image and return the following structured details:

        **Required Output:**
        - **Gender** (Male/Female/Non-binary)
        - **Age Estimate** (e.g., 25 years)
        - **Ethnicity** (e.g., Asian, Caucasian, African, etc.)
        - **Mood** (e.g., Happy, Sad, Neutral, Excited)
        - **Facial Expression** (e.g., Smiling, Frowning, Neutral, etc.)
        - **Glasses** (Yes/No)
        - **Beard** (Yes/No)
        - **Hair Color** (e.g., Black, Blonde, Brown)
        - **Eye Color** (e.g., Blue, Green, Brown)
        - **Headwear** (Yes/No, specify type if applicable)
        - **Emotions Detected** (e.g., Joyful, Focused, Angry, etc.)
        - **Confidence Level** (Accuracy of prediction in percentage)

        Please return structured results clearly. No apologies or empty responses.
        """
        response = model.generate_content([prompt, image])
        return response.text.strip()
    except Exception as e:
        return f"❌ Error: {str(e)}"

# Streamlit UI
st.title("👤 Know A Human 👤")
st.write("Upload an image to detect human attributes using AI.")

# Image Upload
uploaded_image = st.file_uploader("📤 Upload an Image", type=['png', 'jpg', 'jpeg'])

if uploaded_image:
    # Load and resize image to a fixed size
    img = PIL.Image.open(uploaded_image)
    img.thumbnail((300, 300))  # Resize to prevent overflow

    # Create two columns for layout (Image on Left, Output on Right)
    col1, col2 = st.columns([1, 2])  # Left column (image) is smaller than right column (output)

    with col1:
        st.image(img, caption="📸 Uploaded Image", use_container_width=False)  # Fixed size

    with col2:
        # Submit button for analysis
        if st.button("🚀 Submit for Analysis"):
            with st.spinner("⏳ Analyzing image... Please wait."):
                person_info = analyze_human_attributes(img)

            if "Error" in person_info:
                st.error(person_info)
            else:
                with st.expander("📋 **Analysis Report**", expanded=True):
                    st.markdown(person_info)

                # Extract confidence level
                confidence = None
                for line in person_info.split("\n"):
                    if "Confidence Level" in line:
                        confidence = line.split(":")[-1].strip()
                        break

                # Clean and display confidence level
                confidence_cleaned = re.sub(r'[^\d.]', '', confidence)  # Remove non-numeric characters
                if confidence_cleaned:
                    st.progress(float(confidence_cleaned) / 100)
                else:
                    st.error("⚠️ Confidence value is missing or not valid.")

            st.success("✅ Analysis Completed!")
