import streamlit as st
import pickle
import numpy as np

# Set page config
st.set_page_config(
    page_title="Laptop Price Predictor",
    page_icon="💻",
    layout="wide"
)

# Custom CSS styling with dark theme
st.markdown("""
<style>
    /* Main container styling */
    .main {
        padding: 2rem;
        background-color: #1a1a1a;
    }
    
    /* Title styling */
    .title {
        text-align: center;
        padding: 1.5rem;
        background: linear-gradient(to right, #2c3e50, #3498db);
        color: #e0e0e0;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    /* Input field containers */
    .stSelectbox, .stSlider, .stNumberInput {
        background-color: #2d2d2d;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        margin-bottom: 1rem;
    }
    
    /* Button styling */
    .stButton>button {
        width: 100%;
        background-color: #2ecc71;
        color: #e0e0e0;
        padding: 0.8rem;
        font-size: 1.2rem;
        border-radius: 10px;
        border: none;
        box-shadow: 0 2px 4px rgba(0,0,0,0.3);
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.4);
        background-color: #27ae60;
    }
    
    /* Prediction result styling */
    .big-font {
        font-size: 1.5rem;
        font-weight: bold;
        padding: 1rem;
        background: linear-gradient(to right, #2c3e50, #27ae60);
        color: #e0e0e0;
        border-radius: 10px;
        text-align: center;
        margin-top: 2rem;
    }

    /* Text and label styling */
    .stMarkdown, .stSelectbox > div > label, .stSlider > div > label, .stNumberInput > div > label {
        color: #e0e0e0 !important;
    }

    /* Overall app background */
    .stApp {
        background-color: #1a1a1a;
    }

    /* Selectbox styling */
    .stSelectbox > div > div {
        background-color: #2d2d2d !important;
        color: #e0e0e0 !important;
    }

    /* Slider styling */
    .stSlider > div > div > div {
        background-color: #2ecc71 !important;
    }

    /* Title text color */
    h1, h2, h3 {
        color: #e0e0e0 !important;
    }

    /* Input field text */
    input, select, textarea {
        background-color: #2d2d2d !important;
        color: #e0e0e0 !important;
        border: 1px solid #404040 !important;
    }

    /* Divider styling */
    hr {
        border-color: #404040 !important;
    }
</style>
""", unsafe_allow_html=True)

# Header section
st.markdown("""
<div class="title">
    <h1>🚀 Advanced Laptop Price Predictor</h1>
    <p>Find the perfect price for your dream laptop configuration</p>
</div>
""", unsafe_allow_html=True)

# Add decorative divider
st.markdown("<hr style='margin: 2rem 0; opacity: 0.3;'>", unsafe_allow_html=True)

# Original code starts here
# import the model
pipe = pickle.load(open('poppy.pkl','rb'))
df = pickle.load(open('dd.pkl','rb'))

st.title("Choose Your Laptop Configuration")

# brand
company = st.selectbox('Brand',df['Company'].unique())

# type of laptop
type = st.selectbox('Type',df['TypeName'].unique())

# Ram
ram = st.selectbox('RAM(in GB)',[2,4,6,8,12,16,24,32,64])

# weight
weight = st.number_input('Weight of the Laptop')

# Touchscreen
touchscreen = st.selectbox('Touchscreen',['No','Yes'])

# IPS
ips = st.selectbox('IPS',['No','Yes'])

# screen size
screen_size = st.slider('Scrensize in inches', 10.0, 18.0, 13.0)

# resolution
resolution = st.selectbox('Screen Resolution',['1920x1080','1366x768','1600x900','3840x2160','3200x1800','2880x1800','2560x1600','2560x1440','2304x1440'])

#cpu
cpu = st.selectbox('CPU',df['CPU Brand'].unique())

hdd = st.selectbox('HDD(in GB)',[0,128,256,512,1024,2048])

ssd = st.selectbox('SSD(in GB)',[0,8,128,256,512,1024])

gpu = st.selectbox('GPU',df['Gpu Brand'].unique())

os = st.selectbox('OS',df['OS Category'].unique())

if st.button('Predict Price'):
    # query
    ppi = None
    if touchscreen == 'Yes':
        touchscreen = 1
    else:
        touchscreen = 0

    if ips == 'Yes':
        ips = 1
    else:
        ips = 0

    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = ((X_res**2) + (Y_res**2))**0.5/screen_size
    query = np.array([company,type,ram,weight,touchscreen,ips,ppi,cpu,hdd,ssd,gpu,os])

    query = query.reshape(1,12)
    
    # Wrap the prediction in styled div
    predicted_price = str(int(np.exp(pipe.predict(query)[0])))
    st.markdown(f"""
    <div class="big-font">
        The predicted price of this configuration is ₹{predicted_price}
    </div>
    """, unsafe_allow_html=True)

# Footer with dark theme
st.markdown("""
<div style='text-align: center; margin-top: 2rem; padding: 1rem; color: #e0e0e0;'>
    <p>Made with ❤️ by Rohit Yadav</p>
    <p style='font-size: 0.8rem;'>Powered by Machine Learning</p>
</div>
""", unsafe_allow_html=True)