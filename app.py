from pathlib import Path
import pandas as pd
import joblib
import streamlit as st

st.set_page_config(page_title='AutoValue AI | Car Price Predictor', page_icon='🚘', layout='wide')
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / 'models' / 'car_price_model.pkl'
DATA_PATH = BASE_DIR / 'data' / 'car data.csv'

st.markdown('''<style>
.stApp {background-image:linear-gradient(rgba(3,10,25,.70),rgba(3,10,25,.72)),url('https://images.pexels.com/photos/32362622/pexels-photo-32362622.jpeg?auto=compress&cs=tinysrgb&w=1920');background-size:cover;background-position:center;background-attachment:fixed;}
[data-testid='stAppViewContainer'] {background:transparent;}
[data-testid='stHeader'] {background:rgba(0,0,0,0);}
.block-container {max-width:1250px;padding-top:2rem;padding-bottom:3rem;}
section[data-testid='stSidebar'] {background:linear-gradient(180deg,rgba(4,13,32,.94),rgba(8,20,43,.90));border-right:1px solid rgba(255,255,255,.10);}
section[data-testid='stSidebar'] * {color:#f8fafc !important;}
.hero {position:relative;overflow:hidden;padding:2rem 2.5rem;border-radius:28px;background:linear-gradient(135deg,rgba(5,17,48,.96),rgba(19,57,125,.90),rgba(37,99,235,.84));border:1px solid rgba(255,255,255,.16);box-shadow:0 20px 55px rgba(0,0,0,.30);color:white;margin-bottom:1.5rem;}
.hero:after {content:'🚘';position:absolute;right:4%;top:4%;font-size:9rem;opacity:.12;}
.hero h1 {font-size:3.15rem;line-height:1.05;margin:.35rem 0 .65rem;color:white;font-weight:850;}
.hero p {font-size:1.05rem;color:#e0ecff;margin:0;}
.hero-line {width:64px;height:4px;background:#38bdf8;border-radius:10px;margin-top:1.1rem;}
.glass {background:rgba(255,255,255,.90);backdrop-filter:blur(18px);-webkit-backdrop-filter:blur(18px);border:1px solid rgba(255,255,255,.65);border-radius:24px;padding:1.35rem 1.5rem;box-shadow:0 18px 50px rgba(0,0,0,.20);margin-bottom:1.2rem;}
[data-testid='stWidgetLabel'] p {color:#0f172a !important;font-weight:650 !important;}
.stTextInput input,.stNumberInput input,.stSelectbox div[data-baseweb='select']>div {background:rgba(248,250,252,.94) !important;border-radius:12px !important;}
div[data-baseweb='select'] span {color:#0f172a !important;}
.stButton>button {border:0 !important;border-radius:13px !important;min-height:3.1rem !important;font-size:1rem !important;font-weight:800 !important;color:white !important;background:linear-gradient(90deg,#1687ff,#2563eb,#4f46e5) !important;box-shadow:0 12px 28px rgba(37,99,235,.35) !important;}
.result {padding:1.7rem 2rem;border-radius:22px;background:rgba(236,253,245,.95);border:1px solid #86efac;text-align:center;box-shadow:0 14px 35px rgba(0,0,0,.15);}
.price {font-size:3.2rem;font-weight:900;color:#047857;}
.footer {text-align:center;color:#e2e8f0;padding:2rem 0 1rem;text-shadow:0 1px 4px rgba(0,0,0,.55);}
</style>''', unsafe_allow_html=True)

st.markdown('''<div class="hero"><div>AI-POWERED VEHICLE VALUATION</div><h1>🚘 AutoValue AI</h1><p>Estimate the resale value of a used car using machine learning.</p><div class="hero-line"></div></div>''', unsafe_allow_html=True)

if not MODEL_PATH.exists():
    st.error('Model not found. Run `python train_model.py` first.')
    st.stop()
if not DATA_PATH.is_file():
    st.error(f'Dataset not found: {DATA_PATH}')
    st.stop()

model = joblib.load(MODEL_PATH)
df = pd.read_csv(DATA_PATH)

with st.sidebar:
    st.markdown('## 🚘 AutoValue AI')
    st.caption('Car Selling Price Prediction')
    st.divider()
    st.markdown('### How it works')
    st.markdown('1. Enter vehicle details\n2. Random Forest analyzes the features\n3. Get an estimated resale price')
    st.divider()
    st.write('**Algorithm:** Random Forest Regressor')
    st.write('**Dataset:** Car Details from Car Dekho')

st.markdown('<div class="glass"><h2 style="margin:0;color:#0f172a;">🚗 Vehicle Information</h2><p style="color:#475569;margin-top:.4rem;">Enter the details of the used car to get an AI-powered market estimate.</p></div>', unsafe_allow_html=True)

left, right = st.columns(2, gap='large')
with left:
    car_name = st.text_input('Car Name', value='Maruti Swift Dzire VDI', help='Enter the car/model name.')
    year = st.number_input('Manufacturing Year', min_value=1990, max_value=2026, value=2017, step=1)
    kms = st.number_input('Kilometers Driven', min_value=0, max_value=1000000, value=30000, step=1000)
with right:
    fuel = st.selectbox('Fuel Type', sorted(df['fuel'].dropna().astype(str).unique()))
    seller = st.selectbox('Seller Type', sorted(df['seller_type'].dropna().astype(str).unique()))
    transmission = st.selectbox('Transmission', sorted(df['transmission'].dropna().astype(str).unique()))
    owner = st.selectbox('Owner', sorted(df['owner'].dropna().astype(str).unique()))

if st.button('💰  PREDICT SELLING PRICE', type='primary', use_container_width=True):
    age = max(0, 2026 - year)
    input_df = pd.DataFrame([{'Car_Name':car_name,'age':age,'Kms_Driven':kms,'Fuel_Type':fuel,'Seller_Type':seller,'Transmission':transmission,'Owner':owner}])
    prediction = max(0.0, float(model.predict(input_df)[0]))
    st.markdown(f'''<div class="result"><div>ESTIMATED RESALE VALUE</div><div class="price">₹{prediction:,.0f}</div><div>Approximate machine-learning estimate. Actual market value may vary.</div></div>''', unsafe_allow_html=True)
    st.write('')
    a,b,c,d = st.columns(4)
    a.metric('Manufacturing Year', str(year)); b.metric('Distance', f'{kms:,} km'); c.metric('Fuel', fuel); d.metric('Owner', owner)

st.markdown('<div class="footer"><strong>AutoValue AI</strong> — Intelligent Used-Car Valuation<br>Built with Python • Pandas • Scikit-learn • Streamlit</div>', unsafe_allow_html=True)
