st.markdown("""
<style>

/* ===== DARK MEDICAL AI BACKGROUND ===== */
.stApp {
    background:
    linear-gradient(rgba(6, 20, 35, 0.92), rgba(6, 20, 35, 0.92)),
    url("https://images.unsplash.com/photo-1530026186672-2cd00ffc50fe?q=80&w=1600&auto=format&fit=crop");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
    color: #EAF2F8;
}

/* ===== FADE ANIMATION ===== */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(12px); }
    to { opacity: 1; transform: translateY(0); }
}

/* ===== TITLE ===== */
.main-title {
    font-size: 40px;
    font-weight: 700;
    text-align: center;
    color: #5DADE2;
    letter-spacing: 0.5px;
    animation: fadeIn 1s ease-in-out;
}

/* ===== DARK GLASS PANEL ===== */
.glass {
    padding: 26px;
    border-radius: 18px;
    background: rgba(10, 25, 45, 0.75);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(93, 173, 226, 0.15);
    box-shadow: 0 10px 35px rgba(0,0,0,0.6);
    animation: fadeIn 0.8s ease-in-out;
}

/* ===== RESULT BOX ===== */
.result-box {
    padding: 18px;
    border-radius: 12px;
    text-align: center;
    font-size: 20px;
    font-weight: 600;
    animation: fadeIn 0.6s ease-in-out;
}

/* ===== BUTTON ===== */
.stButton > button {
    height: 52px;
    border-radius: 10px;
    font-size: 18px;
    font-weight: 600;
    background: linear-gradient(90deg, #2E86C1, #5DADE2);
    color: white;
    border: none;
}

/* ===== INPUT TEXT VISIBILITY ===== */
label, .stSelectbox, .stNumberInput, .stSlider {
    color: #EAF2F8 !important;
}

</style>
""", unsafe_allow_html=True)
