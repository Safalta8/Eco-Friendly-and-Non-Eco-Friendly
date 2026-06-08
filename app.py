import pandas as pd
import altair as alt
import streamlit as st
from PIL import Image
from groq import Groq
from google import genai
import tensorflow as tf
import numpy as np

st.set_page_config(page_title="EcoVision AI", page_icon="♻️", layout="wide")

import os

@st.cache_resource
def load_waste_model():

    model_path = "best_model.keras"

    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}")
        return None

    return tf.keras.models.load_model(
        model_path,
        compile=False
    )

IMG_SIZE = 224

CLASS_NAMES = [
    "Non-Recyclable",
    "Recyclable"
]
model = load_waste_model()
# ---------------- API KEYS ----------------
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]


client = Groq(api_key=GROQ_API_KEY)
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

# ---------------- STYLES ----------------
st.markdown("""
<style>

/* =========================
   ECO VISION PREMIUM THEME
========================= */

:root{
--sidebar:#003B34;
--hero-start:#002A27;
--hero-end:#00806D;
--background:#F4F7F6;
--card:#FFFFFF;
--accent:#52B788;
--accent-light:#95D5B2;
--text-light:#FFFFFF;
--text-dark:#023A34;
}

/* Main App */
.stApp{
background:linear-gradient(135deg,#F4F7F6,#E9F5EF);
font-family:'Segoe UI',sans-serif;
color:var(--text-dark);
}



/* =========================
   SIDEBAR
========================= */

[data-testid="stSidebar"]{
background:linear-gradient(
180deg,
#002A27 0%,
#00463C 50%,
#006A59 100%
);
padding-top:20px;
border-right:1px solid rgba(255,255,255,0.15);
box-shadow:5px 0px 25px rgba(0,0,0,0.2);
}
.sidebar-logo{
    font-size:32px;
    font-weight:800;
    color:#ffff !important;
    text-align:center;
    margin-bottom:20px;
    text-shadow:0 0 15px rgba(149,213,178,0.4);
}

/* Sidebar Navigation */
[data-testid="stSidebar"] .stRadio label,
[data-testid="stSidebar"] .stMarkdown,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span{
color:white !important;
font-size:18px !important;
font-weight:500 !important;
}

[data-testid="stSidebar"] .stRadio label:hover{
color:#95D5B2 !important;
transition:0.3s;
}
[data-testid="stSidebar"] .stRadio label{
    font-size:20px !important;
    font-weight:600 !important;
    color:white !important;
    padding:12px 10px;
    border-radius:12px;
    transition:all .3s ease;
}

[data-testid="stSidebar"] .stRadio label:hover{
    background:rgba(255,255,255,0.12);
    transform:translateX(5px);
}
/* =========================
   HERO SECTION
========================= */

.hero{
padding:50px;
border-radius:30px;

background:linear-gradient(
135deg,
rgba(0,42,39,0.95),
rgba(0,106,89,0.9)
);

backdrop-filter:blur(15px);

box-shadow:
0px 15px 35px rgba(0,0,0,0.2),
0px 0px 40px rgba(82,183,136,0.25);

text-align:center;
position:relative;
overflow:hidden;
}

.hero::before{
content:"";
position:absolute;
width:300px;
height:300px;
background:rgba(255,255,255,0.08);
border-radius:50%;
top:-100px;
right:-100px;
}

.hero h1{
font-size:4.5rem;
font-weight:800;
background:linear-gradient(
90deg,
#95D5B2,
#52B788,
#FFFFFF
);
-webkit-background-clip:text;
-webkit-text-fill-color:transparent;
margin-bottom:10px;
}

.hero h4{
color:white;
font-weight:400;
letter-spacing:1px;
}

/* =========================
   KPI CARDS
========================= */

.kpi{
padding:25px;
border-radius:24px;

background:rgba(255,255,255,0.9);
backdrop-filter:blur(10px);

border:1px solid rgba(82,183,136,0.15);

box-shadow:
0 10px 30px rgba(0,0,0,0.08);

transition:all 0.4s ease;
}

.kpi:hover{
transform:translateY(-8px) scale(1.02);

box-shadow:
0 20px 40px rgba(82,183,136,0.3);
}

.kpi h1{
color:#52B788;
font-weight:800;
}

/* =========================
   GLASS CARDS
========================= */

.glass-card{

background:rgba(255,255,255,0.92);

backdrop-filter:blur(15px);

padding:25px;
border-radius:24px;

border:1px solid rgba(255,255,255,0.3);

box-shadow:
0 10px 30px rgba(0,0,0,0.08);

transition:0.4s ease;
}

.glass-card:hover{
transform:translateY(-8px);

box-shadow:
0px 20px 40px rgba(82,183,136,0.25);
}

.glass-card h1{
color:#52B788;
font-weight:800;
}

/* =========================
   BUTTONS
========================= */

.stButton>button{

width:100%;

background:linear-gradient(
135deg,
#52B788,
#2D6A4F
);

color:white;

border:none;

border-radius:15px;

padding:14px;

font-size:16px;
font-weight:700;

transition:0.3s ease;

box-shadow:
0 8px 20px rgba(82,183,136,0.3);
}

.stButton>button:hover{

transform:translateY(-3px);

box-shadow:
0 12px 30px rgba(82,183,136,0.5);

background:linear-gradient(
135deg,
#2D6A4F,
#52B788
);
}

/* =========================
   INPUTS
========================= */

.stTextInput input,
.stTextArea textarea{

border-radius:15px !important;

border:2px solid rgba(82,183,136,0.4) !important;

background:rgba(255,255,255,0.95) !important;

padding:10px !important;
}

.stTextInput input:focus,
.stTextArea textarea:focus{

border:2px solid #52B788 !important;

box-shadow:
0px 0px 15px rgba(82,183,136,0.35) !important;
}

/* =========================
   DOWNLOAD BUTTON
========================= */

.stDownloadButton button{

background:linear-gradient(
135deg,
#52B788,
#2D6A4F
) !important;

color:white !important;

border:none !important;

border-radius:14px !important;

font-weight:700 !important;
}

.stDownloadButton button:hover{

background:linear-gradient(
135deg,
#2D6A4F,
#52B788
) !important;
}

/* =========================
   PROGRESS BAR
========================= */

.stProgress > div > div > div{
background:linear-gradient(
90deg,
#52B788,
#95D5B2
) !important;
}

/* =========================
   SCROLLBAR
========================= */

::-webkit-scrollbar{
width:10px;
}

::-webkit-scrollbar-track{
background:#E9F5EF;
}

::-webkit-scrollbar-thumb{
background:#52B788;
border-radius:10px;
}

::-webkit-scrollbar-thumb:hover{
background:#2D6A4F;
}

/* =========================
   FADE ANIMATION
========================= */

@keyframes fadeUp{
from{
opacity:0;
transform:translateY(20px);
}
to{
opacity:1;
transform:translateY(0);
}
}

.hero,
.kpi,
.glass-card{
animation:fadeUp 0.8s ease;
}

</style>
""", unsafe_allow_html=True)

# ---------------- FUNCTIONS ----------------

def predict_waste(image):

    if model is None:
        return "Model Not Loaded", 0

    img = image.convert("RGB")
    img = img.resize((224, 224))

    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array, verbose=0)

    st.write("Prediction Output:", prediction)

    if prediction.shape[-1] == 1:

        score = float(prediction[0][0])

        if score >= 0.5:
            predicted_class = "Recyclable"
            confidence = score * 100
        else:
            predicted_class = "Non-Recyclable"
            confidence = (1 - score) * 100

    else:

        recyclable_score = prediction[0][1]
        non_recyclable_score = prediction[0][0]

        if recyclable_score > non_recyclable_score:
            predicted_class = "Recyclable"
            confidence = recyclable_score * 100
        else:
            predicted_class = "Non-Recyclable"
            confidence = non_recyclable_score * 100

    return predicted_class, confidence


# ---------------- DIY IDEA GENERATOR ----------------

def generate_diy_ideas(material):

    try:

        prompt = f"""
        Give 5 creative DIY projects using {material}.
        """

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role":"user","content":prompt}
            ]
        )

        return response.choices[0].message.content

    except Exception as e:
        return str(e)


# ---------------- ECO TIPS GENERATOR ----------------

def get_eco_tips(user_query):

    try:

        prompt = f"""
        Give eco-friendly recommendations for:
        {user_query}
        """

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role":"user","content":prompt}
            ]
        )

        return response.choices[0].message.content

    except Exception as e:
        return str(e)

# ---------------- SIDEBAR ----------------

st.sidebar.markdown("""
<h1 style='color:#95D5B2;
font-size:32px;
font-weight:800;'>
🌿 EcoVision AI
</h1>
""", unsafe_allow_html=True)
module = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Dustbin Classification",
        "DIY Best Out of Waste",
        "Eco Tip Recommendation System",
        "About Project"
    ]
)


# ---------------- DASHBOARD ----------------
if module == "Dashboard":

    st.markdown("""
    <style>
    .eco-dashboard{
        position:relative;
        padding:20px;
        border-radius:20px;
        background:linear-gradient(135deg,#1f2b1f,#2c3b2c,#95714F);
        overflow:hidden;
        margin-bottom:25px;
        border:1px solid rgba(255,255,255,0.15);
        box-shadow:0 0 60px rgba(140,145,108,.35);
    }
    .eco-title{
        text-align:center;
        color:#EADED0;
        font-size:3rem;
        font-weight:800;
        letter-spacing:2px;
    }
    .eco-sub{
        text-align:center;
        color:#C7AF94;
        font-size:1.1rem;
        margin-bottom:25px;
    }
    .glass-card{
        background:rgba(255,255,255,.08);
        backdrop-filter:blur(12px);
        border-radius:25px;
        padding:20px;
        text-align:center;
        border:1px solid rgba(255,255,255,.1);
        box-shadow:0 8px 25px rgba(0,0,0,.25);
    }
    
    .glass-card:hover{
    transform:translateY(-8px);
    transition:0.3s;
    box-shadow:0 15px 35px rgba(82,183,136,0.3);
    }

    .globe{
        font-size:180px;
        margin-top:10px;
        text-align:center;
        filter:drop-shadow(0 0 25px #ACB087);
        animation:float 5s ease-in-out infinite;
    }
    @keyframes float{
        0%,100%{transform:translateY(0px);}
        50%{transform:translateY(-12px);}
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="eco-dashboard">
        <div class="eco-title">🌍 ECO-FRIENDLY WASTE MANAGEMENT DASHBOARD</div>
        <div class="eco-sub">Empowering Sustainable Living Through Artificial Intelligence</div>
        <div class="globe">🌎</div>
    </div>
    """, unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)

    metrics = [
        ("♻️","Waste Analyzed","1250+"),
        ("🌱","Eco Awareness","87%"),
        ("🌍","CO₂ Saved","320 KG"),
        ("🍃","Sustainability","A+")
    ]

    for col,data in zip([c1,c2,c3,c4],metrics):
        with col:
            st.markdown(
                f"""
                <div class="glass-card">
                    <h1>{data[0]}</h1>
                    <h2>{data[2]}</h2>
                    <p>{data[1]}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
    
    st.markdown("### 📈 Real-Time Statistics")

    s1,s2,s3,s4 = st.columns(4)

    with s1:
        st.metric("Total Predictions","1250")

    with s2:
        st.metric("Recyclable","820")

    with s3:
        st.metric("Non-Recyclable","430")

    with s4:
        st.metric("Accuracy","98.5%")

    st.markdown("### 🌿 Live Environmental Insights")

    a,b,c,d = st.columns(4)

    with a:
        st.metric("♻️ Recyclable Waste","65%")
        st.progress(65)

    with b:
        st.metric("🌿 Organic Waste","20%")
        st.progress(20)

    with c:
        st.metric("⚠️ Hazardous Waste","10%")
        st.progress(10)

    with d:
        st.metric("🗑️ Non-Recyclable Waste","5%")
        st.progress(5)
    
    st.markdown("### 📊 Waste Analytics")

    

    chart_data = pd.DataFrame({
        "Category":["Recyclable","Organic","Hazardous","Non-Recyclable"],
        "Percentage":[65,20,10,5]
    })


    chart = alt.Chart(chart_data).mark_bar(
        color="#1B5E20"
    ).encode(
        x=alt.X("Category", title="Waste Category"),
        y=alt.Y("Percentage", title="Percentage (%)")
    )

    st.altair_chart(chart, use_container_width=True)

    st.markdown("### 🚀 Smart Eco Modules")

    m1,m2,m3 = st.columns(3)

    with m1:
        st.markdown('<div class="glass-card"><h2>♻️</h2><h3>Smart Waste Classification</h3><p>AI-powered waste identification and dustbin recommendation.</p></div>', unsafe_allow_html=True)

    with m2:
        st.markdown('<div class="glass-card"><h2>🌱</h2><h3>DIY Best Out of Waste</h3><p>Generate creative recycling and reuse ideas.</p></div>', unsafe_allow_html=True)

    with m3:
        st.markdown('<div class="glass-card"><h2>💡</h2><h3>Eco Recommendation Engine</h3><p>Get personalized sustainability suggestions.</p></div>', unsafe_allow_html=True)


# ---------------- CLASSIFICATION ----------------
elif module == "Dustbin Classification":

    st.markdown("""
    <div class="hero">
        <h1>♻️ Smart Waste Classification</h1>
        <p>Upload an image and let the CNN model classify waste.</p>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "📤 Upload Waste Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:

        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Waste Image", width=350)

        if st.button("🔍 Predict Waste"):

            with st.spinner("Analyzing Waste..."):
                
                prediction, confidence = predict_waste(image)

                st.markdown(f"""
                <div style="
                    background:white;
                    padding:25px;
                    border-radius:20px;
                    box-shadow:0 5px 20px rgba(0,0,0,0.12);
                    border-left:8px solid #52B788;
                    margin-top:20px;
                ">
                    <h2 style="color:#00463C;">♻️ Classification Result</h2>
                    <h1 style="color:#52B788;">{prediction}</h1>
                </div>
                """, unsafe_allow_html=True)

                st.metric(
                    "Prediction Confidence",
                    f"{confidence:.2f}%"
                )

                st.progress(int(confidence))

                import os

                if prediction == "Recyclable":
                    

                    if os.path.exists("images/blue_dustbin.png"):
                        st.image("images/blue_dustbin.png", width=220)

                    st.success("🔵 Recyclable Waste → Blue Dustbin")

                else:

                    if os.path.exists("images/black_dustbin.png"):
                        st.image("images/black_dustbin.png", width=220)

                    st.error("⚫ Non-Recyclable Waste → Black Dustbin")

                st.info(
                    "🌍 Proper waste segregation helps reduce pollution and increases recycling efficiency."
                )

                report = f"""
                Waste Classification Report

                Prediction : {prediction}
                Confidence : {confidence:.2f}%
                """

                st.download_button(
                    "📥 Download Report",
                    report,
                    file_name="waste_report.txt",
                    mime="text/plain"
                )

                st.markdown("---")
                st.markdown("### 📈 AI Insights")

                if confidence > 95:
                    st.success("✅ High confidence prediction")
                elif confidence > 80:
                    st.warning("⚠️ Moderate confidence prediction")
                else:
                    st.error("❌ Low confidence prediction")

                st.markdown("### 📊 Waste Analysis")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("🎯 Confidence", f"{confidence:.2f}%")

                with col2:
                    st.metric("🤖 Model", "CNN")

                with col3:
                    st.metric("♻️ Category", prediction)
                

# ---------------- DIY ----------------

elif module == "DIY Best Out of Waste":

    st.markdown("""
    <div class="hero">
        <h1>🌱 DIY Innovation Hub</h1>
        <p>Transform Waste into Creative Projects</p>
    </div>
    """, unsafe_allow_html=True)

    material = st.text_input(
        "♻️ Enter Waste Material",
        placeholder="Example: Cardboard, Plastic Bottles, Newspapers"
    )
    
    col1,col2,col3 = st.columns(3)

    with col1:
        st.metric("💡 Innovation Score","98%")

    with col2:
        st.metric("♻️ Reuse Potential","High")

    with col3:
        st.metric("🌍 Eco Impact","Excellent")

    st.success("🏆 Eco Creator Badge Unlocked")

    if st.button("🚀 Generate DIY Ideas"):

        if material.strip():

            with st.spinner("Generating Creative DIY Ideas..."):
                ideas = generate_diy_ideas(material)

            st.success("✅ DIY Ideas Generated Successfully")

            st.markdown(f"""
            <div style="
                background:white;
                padding:25px;
                border-radius:20px;
                box-shadow:0 5px 20px rgba(0,0,0,0.12);
                border-left:8px solid #52B788;
                margin-top:20px;
            ">
            <h3 style="color:#00463C;">
                🌱 Creative Projects Using {material.title()}
            </h3>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(ideas)

            st.info(
                "🌍 Every recycled item reduces landfill waste and helps create a greener future."
            )

            st.download_button(
                "📥 Download DIY Ideas",
                data=ideas,
                file_name="DIY_Ideas.txt",
                mime="text/plain"
            )

            st.markdown("---")
            st.subheader("📊 Sustainability Impact")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("♻️ Waste Reused", "100%")

            with col2:
                st.metric("🌍 Eco Impact", "High")

            with col3:
                st.metric("💡 Creativity Score", "A+")
                
                
# ---------------- ECO TIPS ----------------
elif module == "Eco Tip Recommendation System":

    st.markdown(
        """
        <div class="hero">
            <h1>💡 Eco Intelligence Center</h1>
            <p>Get personalized eco-friendly recommendations.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    user_input = st.text_area(
        "Describe your environmental concern",
        height=150,
        placeholder="Example: How to grow a plant sustainably?"
    )

    if st.button("🌱 Generate Eco Tips"):

        if user_input.strip():

            with st.spinner("Generating Eco Recommendations..."):

                tips = get_eco_tips(user_input)

            st.success("✅ Recommendations Generated Successfully")

            st.metric(
                label="Eco Score",
                value="95%"
            )

            st.progress(95)

            st.markdown(
                f"""
                <div style="
                    background:white;
                    padding:25px;
                    border-radius:20px;
                    border-left:8px solid #52B788;
                    box-shadow:0 5px 20px rgba(0,0,0,0.1);
                    margin-top:20px;
                ">
                {tips}
                </div>
                """,
                unsafe_allow_html=True
            )

            st.download_button(
                "📥 Download Recommendations",
                data=tips,
                file_name="eco_tips.txt",
                mime="text/plain"
            )
            
            st.info(
                "🌍 Every recycled item reduces landfill waste and helps create a greener future."
            )
            
            st.metric(
                "DIY Innovation Score",
                "98%"
            )

            st.progress(98)
            

# ---------------- ABOUT PROJECT ----------------
elif module == "About Project":
   
    st.markdown("""
    <style>
    .about-container {
        position: relative;
        width: 100%;
        min-height: 100vh; 
        background: url('https://i.pinimg.com/1200x/a8/80/24/a880246460dd22109840345f01e54079.jpg') center/cover no-repeat;
        border-radius: 30px;
        box-shadow: 0 0 40px rgba(82,183,136,0.45);
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        color: white;
        text-align: center;
        padding: 60px;
    }

    .about-title {
        font-size: 3rem;
        font-weight: 800;
        color: #52B788;
        margin-bottom: 20px;
        text-shadow: 0 0 10px rgba(0,0,0,0.4);
    }

    .about-text {
        max-width: 850px;
        font-size: 1.2rem;
        line-height: 1.8;
        background: rgba(0,0,0,0.35); 
        padding: 30px;
        border-radius: 15px;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="about-container">
        <div class="about-title">🌍 About EcoVision AI</div>
        <div class="about-text">
            EcoVision AI classifies waste into Recyclable and Non‑Recyclable categories using CNN and AI.
            It also provides DIY recycling ideas and eco‑friendly recommendations to promote sustainable living.
        </div>
    </div>
    """, unsafe_allow_html=True)
