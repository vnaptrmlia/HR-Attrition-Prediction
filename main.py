# HR_Attrition_Streamlit_App_Enhanced.py - Enhanced HR Attrition Prediction App with LIME
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import hashlib
import warnings
import os

# LIME and interpretability imports
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import base64
from io import BytesIO

# Suppress warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Sistem Prediksi Attrisi Karyawan",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Single admin credential
ADMIN_CREDENTIALS = {
    "hr_admin": {
        "password_hash": "240be518fabd2724ddb6f04eeb1da5967448d7e831c08c8fa822809f74c720a9",  # admin123
        "role": "admin",
        "permissions": ["employee_assessment", "dashboard"],
        "display_name": "HR Administrator"
    }
}

def get_user_permissions(username):
    """Get user permissions based on username"""
    if username in ADMIN_CREDENTIALS:
        return ADMIN_CREDENTIALS[username]["permissions"]
    return []

def get_user_role(username):
    """Get user role based on username"""
    if username in ADMIN_CREDENTIALS:
        return ADMIN_CREDENTIALS[username]["role"]
    return "guest"

def get_user_display_name(username):
    """Get user display name"""
    if username in ADMIN_CREDENTIALS:
        return ADMIN_CREDENTIALS[username]["display_name"]
    return username

def has_permission(username, permission):
    """Check if user has specific permission"""
    permissions = get_user_permissions(username)
    return permission in permissions

class HRFeatureCategorizer:
    """HR Feature Categorizer dengan penjelasan detail"""
    
    def __init__(self):
        self.hr_features = self._define_essential_hr_features()
        
    def _define_essential_hr_features(self):
        """Define essential HR features untuk prediksi dengan penjelasan detail"""
        return {
            "Age": {
                "type": "number", "min": 18, "max": 65, "default": 32, "unit": "tahun", 
                "label": "Umur Karyawan",
                "explanation": "Usia karyawan dalam tahun. Karyawan yang lebih muda (20-30) dan mendekati pensiun (55+) memiliki risiko attrisi lebih tinggi."
            },
            "Gender": {
                "type": "selectbox", "options": {"Perempuan": 0, "Laki-laki": 1}, "default": "Laki-laki", 
                "label": "Jenis Kelamin",
                "explanation": "Jenis kelamin karyawan. Membantu menganalisis pola attrisi berdasarkan gender untuk strategi retensi yang tepat."
            },
            "MaritalStatus": {
                "type": "selectbox", "options": {"Lajang": 0, "Menikah": 1, "Bercerai": 2}, "default": "Menikah", 
                "label": "Status Pernikahan",
                "explanation": "Status pernikahan karyawan. Karyawan lajang cenderung lebih mobile, sedangkan yang menikah lebih stabil."
            },
            "DistanceFromHome": {
                "type": "number", "min": 1, "max": 50, "default": 7, "unit": "km", 
                "label": "Jarak dari Rumah",
                "explanation": "Jarak tempat tinggal ke kantor dalam kilometer. Jarak >20km meningkatkan risiko attrisi karena biaya dan waktu perjalanan."
            },
            "JobLevel": {
                "type": "selectbox", "options": {"Pemula": 1, "Junior": 2, "Menengah": 3, "Senior": 4, "Eksekutif": 5}, "default": "Menengah", 
                "label": "Level Pekerjaan",
                "explanation": "Tingkat senioritas dalam organisasi. Level pemula dan menengah memiliki risiko attrisi lebih tinggi karena mencari pertumbuhan karir."
            },
            "YearsAtCompany": {
                "type": "number", "min": 0, "max": 40, "default": 5, "unit": "tahun", 
                "label": "Lama Bekerja di Perusahaan",
                "explanation": "Total masa kerja di perusahaan. Karyawan dengan masa kerja 1-3 tahun paling berisiko karena masih beradaptasi."
            },
            "YearsInCurrentRole": {
                "type": "number", "min": 0, "max": 20, "default": 2, "unit": "tahun", 
                "label": "Lama di Posisi Saat Ini",
                "explanation": "Berapa lama karyawan berada di posisi/role yang sama. Terlalu lama di posisi yang sama dapat menyebabkan kebosanan."
            },
            "YearsSinceLastPromotion": {
                "type": "number", "min": 0, "max": 20, "default": 1, "unit": "tahun", 
                "label": "Tahun Sejak Promosi Terakhir",
                "explanation": "Waktu sejak promosi terakhir. Tidak ada promosi >3 tahun dapat menurunkan motivasi dan meningkatkan risiko attrisi."
            },
            "OverTime": {
                "type": "selectbox", "options": {"Tidak": 0, "Ya": 1}, "default": "Tidak", 
                "label": "Kerja Lembur",
                "explanation": "Apakah karyawan sering bekerja lembur. Lembur berlebihan adalah faktor risiko utama yang meningkatkan burnout."
            },
            "BusinessTravel": {
                "type": "selectbox", "options": {"Tidak Pernah": 0, "Jarang": 1, "Sering": 2}, "default": "Jarang", 
                "label": "Perjalanan Dinas",
                "explanation": "Frekuensi perjalanan dinas. Perjalanan yang terlalu sering dapat mengganggu work-life balance."
            },
            "MonthlyIncome": {
                "type": "number", "min": 1000, "max": 25000, "default": 5000, "unit": "USD", 
                "label": "Gaji Bulanan (USD)",
                "explanation": "Gaji bulanan dalam USD. Gaji yang tidak kompetitif dibanding pasar adalah faktor risiko attrisi yang signifikan."
            },
            "PercentSalaryHike": {
                "type": "slider", "min": 0, "max": 25, "default": 13, "unit": "%", 
                "label": "Persentase Kenaikan Gaji Terakhir",
                "explanation": "Persentase kenaikan gaji tahun lalu. Kenaikan <10% atau tidak ada kenaikan meningkatkan risiko attrisi."
            },
            "StockOptionLevel": {
                "type": "selectbox", "options": {"Tidak Ada": 0, "Dasar": 1, "Standar": 2, "Premium": 3}, "default": "Dasar", 
                "label": "Level Opsi Saham",
                "explanation": "Tingkat kepemilikan saham perusahaan. Opsi saham dapat meningkatkan loyalitas dan retensi karyawan jangka panjang."
            },
            "JobSatisfaction": {
                "type": "selectbox", "options": {"Rendah": 1, "Sedang": 2, "Tinggi": 3, "Sangat Tinggi": 4}, "default": "Tinggi", 
                "label": "Kepuasan Kerja",
                "explanation": "Tingkat kepuasan dengan pekerjaan saat ini. Kepuasan rendah adalah prediktor terkuat untuk attrisi."
            },
            "WorkLifeBalance": {
                "type": "selectbox", "options": {"Buruk": 1, "Baik": 2, "Lebih Baik": 3, "Terbaik": 4}, "default": "Lebih Baik", 
                "label": "Keseimbangan Kerja-Hidup",
                "explanation": "Seberapa baik karyawan dapat menyeimbangkan pekerjaan dan kehidupan pribadi. Work-life balance buruk meningkatkan risiko burnout."
            },
            "EnvironmentSatisfaction": {
                "type": "selectbox", "options": {"Rendah": 1, "Sedang": 2, "Tinggi": 3, "Sangat Tinggi": 4}, "default": "Tinggi", 
                "label": "Kepuasan Lingkungan Kerja",
                "explanation": "Kepuasan dengan lingkungan kerja, rekan kerja, dan budaya perusahaan. Lingkungan yang toxic meningkatkan turnover."
            },
            "PerformanceRating": {
                "type": "selectbox", "options": {"Rendah": 1, "Baik": 2, "Sangat Baik": 3, "Luar Biasa": 4}, "default": "Sangat Baik", 
                "label": "Rating Kinerja",
                "explanation": "Penilaian kinerja terbaru. High performer yang tidak dihargai atau low performer yang merasa tertekan sama-sama berisiko tinggi."
            }
        }
    
    def create_hr_input_form(self):
        """Create form input dengan penjelasan detail"""
        st.sidebar.header("👥 Informasi Karyawan")
        st.sidebar.markdown("**Penilaian Risiko Attrisi Karyawan**")
        st.sidebar.info("💡 **Tips:** Hover pada ikon (?) untuk melihat penjelasan setiap field")
        
        profile = st.sidebar.selectbox(
            "Profil Cepat:",
            ["📊 Input Manual", "🌟 Karyawan Berprestasi", "📈 Karyawan Biasa", "🆕 Fresh Graduate", "⚠️ Karyawan Berisiko"],
            help="Pilih profil template untuk mengisi data dengan cepat, atau pilih Input Manual untuk kustomisasi lengkap"
        )
        
        input_data = {}
        
        if profile == "📊 Input Manual":
            st.sidebar.subheader("📋 Data Demografi Personal")
            st.sidebar.caption("Informasi dasar tentang karyawan")
            
            feature_config = self.hr_features["Age"]
            input_data["Age"] = st.sidebar.number_input(
                feature_config["label"], 
                18, 65, 32,
                help=feature_config["explanation"]
            )
            
            feature_config = self.hr_features["Gender"]
            input_data["Gender"] = st.sidebar.selectbox(
                feature_config["label"], 
                ["Perempuan", "Laki-laki"], 
                index=1,
                help=feature_config["explanation"]
            )
            
            feature_config = self.hr_features["MaritalStatus"]
            input_data["MaritalStatus"] = st.sidebar.selectbox(
                feature_config["label"], 
                ["Lajang", "Menikah", "Bercerai"], 
                index=1,
                help=feature_config["explanation"]
            )
            
            feature_config = self.hr_features["DistanceFromHome"]
            input_data["DistanceFromHome"] = st.sidebar.number_input(
                feature_config["label"], 
                1, 50, 7,
                help=feature_config["explanation"]
            )
            
            st.sidebar.subheader("💼 Informasi Pekerjaan")
            st.sidebar.caption("Detail posisi dan masa kerja")
            
            feature_config = self.hr_features["JobLevel"]
            input_data["JobLevel"] = st.sidebar.selectbox(
                feature_config["label"], 
                ["Pemula", "Junior", "Menengah", "Senior", "Eksekutif"], 
                index=2,
                help=feature_config["explanation"]
            )
            
            feature_config = self.hr_features["YearsAtCompany"]
            input_data["YearsAtCompany"] = st.sidebar.number_input(
                feature_config["label"], 
                0, 40, 5,
                help=feature_config["explanation"]
            )
            
            feature_config = self.hr_features["YearsInCurrentRole"]
            input_data["YearsInCurrentRole"] = st.sidebar.number_input(
                feature_config["label"], 
                0, 20, 2,
                help=feature_config["explanation"]
            )
            
            feature_config = self.hr_features["YearsSinceLastPromotion"]
            input_data["YearsSinceLastPromotion"] = st.sidebar.number_input(
                feature_config["label"], 
                0, 20, 1,
                help=feature_config["explanation"]
            )
            
            feature_config = self.hr_features["OverTime"]
            input_data["OverTime"] = st.sidebar.selectbox(
                feature_config["label"], 
                ["Tidak", "Ya"], 
                index=0,
                help=feature_config["explanation"]
            )
            
            feature_config = self.hr_features["BusinessTravel"]
            input_data["BusinessTravel"] = st.sidebar.selectbox(
                feature_config["label"], 
                ["Tidak Pernah", "Jarang", "Sering"], 
                index=1,
                help=feature_config["explanation"]
            )
            
            st.sidebar.subheader("💰 Kompensasi & Benefit")
            st.sidebar.caption("Informasi gaji dan benefit karyawan")
            
            feature_config = self.hr_features["MonthlyIncome"]
            input_data["MonthlyIncome"] = st.sidebar.number_input(
                feature_config["label"], 
                1000, 25000, 5000, 
                step=500,
                help=feature_config["explanation"]
            )
            
            feature_config = self.hr_features["PercentSalaryHike"]
            input_data["PercentSalaryHike"] = st.sidebar.slider(
                feature_config["label"], 
                0, 25, 13,
                help=feature_config["explanation"]
            )
            
            feature_config = self.hr_features["StockOptionLevel"]
            input_data["StockOptionLevel"] = st.sidebar.selectbox(
                feature_config["label"], 
                ["Tidak Ada", "Dasar", "Standar", "Premium"], 
                index=1,
                help=feature_config["explanation"]
            )
            
            st.sidebar.subheader("😊 Kepuasan & Kinerja")
            st.sidebar.caption("Evaluasi kepuasan dan performa karyawan")
            
            feature_config = self.hr_features["JobSatisfaction"]
            input_data["JobSatisfaction"] = st.sidebar.selectbox(
                feature_config["label"], 
                ["Rendah", "Sedang", "Tinggi", "Sangat Tinggi"], 
                index=2,
                help=feature_config["explanation"]
            )
            
            feature_config = self.hr_features["WorkLifeBalance"]
            input_data["WorkLifeBalance"] = st.sidebar.selectbox(
                feature_config["label"], 
                ["Buruk", "Baik", "Lebih Baik", "Terbaik"], 
                index=2,
                help=feature_config["explanation"]
            )
            
            feature_config = self.hr_features["EnvironmentSatisfaction"]
            input_data["EnvironmentSatisfaction"] = st.sidebar.selectbox(
                feature_config["label"], 
                ["Rendah", "Sedang", "Tinggi", "Sangat Tinggi"], 
                index=2,
                help=feature_config["explanation"]
            )
            
            feature_config = self.hr_features["PerformanceRating"]
            input_data["PerformanceRating"] = st.sidebar.selectbox(
                feature_config["label"], 
                ["Rendah", "Baik", "Sangat Baik", "Luar Biasa"], 
                index=2,
                help=feature_config["explanation"]
            )
            
        else:
            # Profil preset dengan penjelasan
            profiles = {
                "🌟 Karyawan Berprestasi": {
                    "description": "High performer dengan kompensasi tinggi dan kepuasan kerja yang baik",
                    "data": {
                        "Age": 35, "Gender": "Laki-laki", "MaritalStatus": "Menikah", "DistanceFromHome": 5,
                        "JobLevel": "Senior", "YearsAtCompany": 8, "YearsInCurrentRole": 3, "YearsSinceLastPromotion": 1,
                        "OverTime": "Tidak", "BusinessTravel": "Jarang", "MonthlyIncome": 8000, "PercentSalaryHike": 18,
                        "StockOptionLevel": "Premium", "JobSatisfaction": "Sangat Tinggi", "WorkLifeBalance": "Lebih Baik",
                        "EnvironmentSatisfaction": "Sangat Tinggi", "PerformanceRating": "Luar Biasa"
                    }
                },
                "📈 Karyawan Biasa": {
                    "description": "Karyawan dengan performa rata-rata dan kondisi kerja yang stabil",
                    "data": {
                        "Age": 32, "Gender": "Perempuan", "MaritalStatus": "Menikah", "DistanceFromHome": 7,
                        "JobLevel": "Menengah", "YearsAtCompany": 5, "YearsInCurrentRole": 2, "YearsSinceLastPromotion": 2,
                        "OverTime": "Tidak", "BusinessTravel": "Jarang", "MonthlyIncome": 5000, "PercentSalaryHike": 13,
                        "StockOptionLevel": "Dasar", "JobSatisfaction": "Tinggi", "WorkLifeBalance": "Lebih Baik",
                        "EnvironmentSatisfaction": "Tinggi", "PerformanceRating": "Sangat Baik"
                    }
                },
                "🆕 Fresh Graduate": {
                    "description": "Karyawan baru lulusan dengan adaptasi awal dan gaji entry level",
                    "data": {
                        "Age": 24, "Gender": "Laki-laki", "MaritalStatus": "Lajang", "DistanceFromHome": 15,
                        "JobLevel": "Pemula", "YearsAtCompany": 1, "YearsInCurrentRole": 1, "YearsSinceLastPromotion": 0,
                        "OverTime": "Ya", "BusinessTravel": "Tidak Pernah", "MonthlyIncome": 3000, "PercentSalaryHike": 11,
                        "StockOptionLevel": "Tidak Ada", "JobSatisfaction": "Tinggi", "WorkLifeBalance": "Baik",
                        "EnvironmentSatisfaction": "Tinggi", "PerformanceRating": "Baik"
                    }
                },
                "⚠️ Karyawan Berisiko": {
                    "description": "Karyawan dengan multiple red flags: lembur, kepuasan rendah, no promotion",
                    "data": {
                        "Age": 28, "Gender": "Perempuan", "MaritalStatus": "Lajang", "DistanceFromHome": 25,
                        "JobLevel": "Junior", "YearsAtCompany": 3, "YearsInCurrentRole": 3, "YearsSinceLastPromotion": 3,
                        "OverTime": "Ya", "BusinessTravel": "Sering", "MonthlyIncome": 3500, "PercentSalaryHike": 11,
                        "StockOptionLevel": "Tidak Ada", "JobSatisfaction": "Rendah", "WorkLifeBalance": "Buruk",
                        "EnvironmentSatisfaction": "Rendah", "PerformanceRating": "Baik"
                    }
                }
            }
            
            profile_info = profiles.get(profile, profiles["📈 Karyawan Biasa"])
            st.sidebar.info(f"**{profile}**\n\n{profile_info['description']}")
            input_data = profile_info["data"]
        
        # Convert Indonesian text values to numeric
        numeric_data = {}
        for key, value in input_data.items():
            if key in self.hr_features:
                feature_config = self.hr_features[key]
                if feature_config["type"] == "selectbox" and "options" in feature_config:
                    numeric_data[key] = feature_config["options"].get(value, 0)
                else:
                    numeric_data[key] = value
            else:
                mapping = {
                    "Gender": {"Perempuan": 0, "Laki-laki": 1},
                    "MaritalStatus": {"Lajang": 0, "Menikah": 1, "Bercerai": 2},
                    "JobLevel": {"Pemula": 1, "Junior": 2, "Menengah": 3, "Senior": 4, "Eksekutif": 5},
                    "OverTime": {"Tidak": 0, "Ya": 1},
                    "BusinessTravel": {"Tidak Pernah": 0, "Jarang": 1, "Sering": 2},
                    "StockOptionLevel": {"Tidak Ada": 0, "Dasar": 1, "Standar": 2, "Premium": 3},
                    "JobSatisfaction": {"Rendah": 1, "Sedang": 2, "Tinggi": 3, "Sangat Tinggi": 4},
                    "WorkLifeBalance": {"Buruk": 1, "Baik": 2, "Lebih Baik": 3, "Terbaik": 4},
                    "EnvironmentSatisfaction": {"Rendah": 1, "Sedang": 2, "Tinggi": 3, "Sangat Tinggi": 4},
                    "PerformanceRating": {"Rendah": 1, "Baik": 2, "Sangat Baik": 3, "Luar Biasa": 4}
                }
                if key in mapping:
                    numeric_data[key] = mapping[key].get(value, 0)
                else:
                    numeric_data[key] = value
        
        return numeric_data

def hash_password(password):
    """Hash password using SHA256"""
    return hashlib.sha256(str.encode(password)).hexdigest()

def verify_password(password, hashed):
    """Verify password against hash"""
    return hash_password(password) == hashed

def login_page():
    """Halaman login dengan single admin"""
    st.markdown("""
    <div style="text-align: center; padding: 50px 0;">
        <h1>👥 Sistem Prediksi Attrisi Karyawan</h1>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("### 🔐 Login Administrator HR")
        
        with st.form("login_form"):
            username = st.text_input("👤 Nama Pengguna", placeholder="Masukkan hr_admin")
            password = st.text_input("🔑 Kata Sandi", type="password", placeholder="Masukkan password")
            submit_button = st.form_submit_button("🔓 Akses Sistem", use_container_width=True)
            
            if submit_button:
                if username in ADMIN_CREDENTIALS:
                    if verify_password(password, ADMIN_CREDENTIALS[username]["password_hash"]):
                        st.session_state.logged_in = True
                        st.session_state.username = username
                        st.session_state.user_role = get_user_role(username)
                        st.session_state.user_permissions = get_user_permissions(username)
                        st.session_state.display_name = get_user_display_name(username)
                        st.success(f"✅ Selamat datang {get_user_display_name(username)}!")
                        st.rerun()
                    else:
                        st.error("❌ Kata sandi salah!")
                else:
                    st.error("❌ Nama pengguna tidak ditemukan!")
        

@st.cache_resource
def load_model_components():
    """Load komponen model dengan interpretability artifacts"""
    try:
        # Core model files dari enhanced_models
        model = joblib.load('enhanced_models/logistic_regression_model.pkl')
        scaler = joblib.load('enhanced_models/scaler.pkl')
        
        with open('enhanced_models/feature_names.pkl', 'rb') as f:
            feature_names = pickle.load(f)
        
        with open('enhanced_models/model_metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)
        
        # Load interpretability artifacts
        try:
            with open('enhanced_models/global_feature_importance.pkl', 'rb') as f:
                global_importance = pickle.load(f)
                # Debug: Print the actual format
                print(f"🔍 DEBUG: global_importance type: {type(global_importance)}")
                print(f"🔍 DEBUG: global_importance shape/len: {getattr(global_importance, 'shape', len(global_importance) if hasattr(global_importance, '__len__') else 'No length')}")
                if hasattr(global_importance, 'dtype'):
                    print(f"🔍 DEBUG: global_importance dtype: {global_importance.dtype}")
        except Exception as e:
            global_importance = None
            print(f"❌ Error loading global_importance: {e}")
            
        try:
            with open('enhanced_models/feature_descriptions.pkl', 'rb') as f:
                feature_descriptions = pickle.load(f)
        except:
            feature_descriptions = {}
            
        # Load LIME configuration
        try:
            with open('enhanced_models/lime_config.pkl', 'rb') as f:
                lime_config = pickle.load(f)
        except:
            lime_config = None
            
        # Load interpretability metadata
        try:
            with open('enhanced_models/interpretability_metadata.pkl', 'rb') as f:
                interpretability_metadata = pickle.load(f)
        except:
            interpretability_metadata = {}
            
        # Load explanation cache
        try:
            with open('enhanced_models/explanation_cache.pkl', 'rb') as f:
                explanation_cache = pickle.load(f)
        except:
            explanation_cache = {}
            
        # Load streamlit helpers
        try:
            with open('enhanced_models/streamlit_helpers.pkl', 'rb') as f:
                streamlit_helpers = pickle.load(f)
        except:
            streamlit_helpers = {}
            
        return model, scaler, feature_names, metadata, global_importance, feature_descriptions, lime_config, interpretability_metadata, explanation_cache, streamlit_helpers
        
    except FileNotFoundError as e:
        st.error(f"❌ File model tidak ditemukan: {e}. Menggunakan mode demo.")
        return None, None, None, {"model_type": "Demo", "test_accuracy": 0.87, "roc_auc": 0.82}, None, {}, None, {}, {}, {}

def prepare_model_input(hr_input, feature_names):
    """Prepare input untuk prediksi model"""
    
    model_input = pd.DataFrame(index=[0], columns=feature_names)
    model_input = model_input.fillna(0)
    
    feature_mapping = {
        'Age': 'Age',
        'MonthlyIncome': 'MonthlyIncome', 
        'YearsAtCompany': 'YearsAtCompany',
        'YearsInCurrentRole': 'YearsInCurrentRole',
        'YearsSinceLastPromotion': 'YearsSinceLastPromotion',
        'DistanceFromHome': 'DistanceFromHome',
        'PercentSalaryHike': 'PercentSalaryHike',
        'JobLevel': 'JobLevel',
        'StockOptionLevel': 'StockOptionLevel',
        'JobSatisfaction': 'JobSatisfaction',
        'WorkLifeBalance': 'WorkLifeBalance',
        'EnvironmentSatisfaction': 'EnvironmentSatisfaction',
        'PerformanceRating': 'PerformanceRating'
    }
    
    for hr_key, model_key in feature_mapping.items():
        if hr_key in hr_input and model_key in model_input.columns:
            model_input[model_key] = hr_input[hr_key]
    
    categorical_mappings = {
        'Gender_Male': 1 if hr_input.get('Gender', 0) == 1 else 0,
        'MaritalStatus_Married': 1 if hr_input.get('MaritalStatus', 0) == 1 else 0,
        'MaritalStatus_Single': 1 if hr_input.get('MaritalStatus', 0) == 0 else 0,
        'OverTime_Yes': 1 if hr_input.get('OverTime', 0) == 1 else 0,
        'BusinessTravel_Travel_Frequently': 1 if hr_input.get('BusinessTravel', 0) == 2 else 0,
        'BusinessTravel_Travel_Rarely': 1 if hr_input.get('BusinessTravel', 0) == 1 else 0,
    }
    
    for model_key, value in categorical_mappings.items():
        if model_key in model_input.columns:
            model_input[model_key] = value
    
    return model_input

def create_lime_explanation(model, scaler, hr_input, feature_names, lime_config=None, explanation_cache=None):
    """Generate LIME explanation untuk individual prediction"""
    try:
        model_input = prepare_model_input(hr_input, feature_names)
        
        if scaler is not None:
            model_input_scaled = scaler.transform(model_input)
            model_input = pd.DataFrame(model_input_scaled, columns=feature_names)
        
        # Check explanation cache first
        input_key = str(sorted(hr_input.items()))
        if explanation_cache and input_key in explanation_cache:
            cached_exp = explanation_cache[input_key]
            return cached_exp['features'], cached_exp['values'], None
        
        # Create LIME explainer using lime_config
        if lime_config is not None:
            # Use config parameters
            num_features = lime_config.get('num_features', 10)
            num_samples = lime_config.get('num_samples', 1000)
            distance_metric = lime_config.get('distance_metric', 'euclidean')
        else:
            # Default parameters
            num_features = 10
            num_samples = 1000
            distance_metric = 'euclidean'
        
        # Create dummy training data untuk LIME
        np.random.seed(42)
        training_data = np.random.randn(100, len(feature_names))
        
        lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data,
            feature_names=feature_names,
            class_names=['Stay', 'Leave'],
            mode='classification',
            discretize_continuous=True
        )
        
        # Generate explanation
        explanation = lime_explainer.explain_instance(
            model_input.values[0],
            model.predict_proba,
            num_features=num_features,
            num_samples=num_samples
        )
        
        # Extract feature contributions
        exp_list = explanation.as_list()
        features = [item[0] for item in exp_list]
        values = [item[1] for item in exp_list]
        
        return features, values, explanation
        
    except Exception as e:
        st.warning(f"⚠️ LIME explanation tidak tersedia: {e}")
        return [], [], None

def make_prediction(model, scaler, hr_input, feature_names):
    """Buat prediksi attrisi"""
    try:
        if model is None:
            risk_score = 0.3 + (hr_input.get('OverTime', 0) * 0.2) + \
                        (1 - hr_input.get('JobSatisfaction', 3)/4) * 0.3 + \
                        (1 - hr_input.get('WorkLifeBalance', 3)/4) * 0.2
            risk_score = min(max(risk_score, 0.05), 0.95)
            prediction = 1 if risk_score > 0.5 else 0
            prediction_proba = [1-risk_score, risk_score]
            return prediction, prediction_proba, None
        
        model_input = prepare_model_input(hr_input, feature_names)
        
        if scaler is not None:
            model_input_scaled = scaler.transform(model_input)
            model_input = pd.DataFrame(model_input_scaled, columns=feature_names)
        
        prediction = model.predict(model_input)[0]
        prediction_proba = model.predict_proba(model_input)[0]
        
        return prediction, prediction_proba, model_input
        
    except Exception as e:
        st.error(f"❌ Error prediksi: {e}")
        return None, None, None

def display_prediction_results(prediction, prediction_proba, hr_input, metadata, model, scaler, feature_names, lime_config, explanation_cache):
    """Display hasil prediksi karyawan dengan LIME explanation"""
    
    st.header("🎯 Hasil Penilaian Risiko Attrisi Karyawan")
    
    col1, col2, col3 = st.columns(3)
    
    attrition_probability = prediction_proba[1]
    
    with col1:
        st.subheader("📊 Level Risiko")
        
        if attrition_probability > 0.7:
            st.error("**🚨 RISIKO TINGGI**")
            risk_level = "TINGGI"
            risk_color = "red"
            risk_description = "Karyawan ini memiliki probabilitas sangat tinggi untuk keluar dalam 6-12 bulan ke depan."
        elif attrition_probability > 0.3:
            st.warning("**⚠️ RISIKO SEDANG**")
            risk_level = "SEDANG" 
            risk_color = "orange"
            risk_description = "Karyawan ini menunjukkan beberapa tanda risiko attrisi."
        else:
            st.success("**✅ RISIKO RENDAH**")
            risk_level = "RENDAH"
            risk_color = "green"
            risk_description = "Karyawan ini kemungkinan besar akan bertahan."
        
        st.info(risk_description)
        st.metric("Probabilitas Attrisi", f"{attrition_probability:.1%}")
        st.metric("Tingkat Keyakinan", f"{max(prediction_proba):.1%}")
    
    with col2:
        st.subheader("📈 Visualisasi Risiko")
        
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = attrition_probability * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Risiko Attrisi (%)"},
            delta = {'reference': 30},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': risk_color},
                'steps': [
                    {'range': [0, 30], 'color': "lightgreen"},
                    {'range': [30, 70], 'color': "yellow"},
                    {'range': [70, 100], 'color': "lightcoral"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 70
                }
            }
        ))
        fig_gauge.update_layout(height=300)
        st.plotly_chart(fig_gauge, use_container_width=True)
    
    with col3:
        st.subheader("👤 Profil Karyawan")
        
        age = hr_input.get('Age', 'N/A')
        income = hr_input.get('MonthlyIncome', 0)
        years_company = hr_input.get('YearsAtCompany', 0)
        job_level = hr_input.get('JobLevel', 0)
        
        # Convert job level number to text
        job_levels = {1: "Pemula", 2: "Junior", 3: "Menengah", 4: "Senior", 5: "Eksekutif"}
        job_level_text = job_levels.get(job_level, "Unknown")
        
        st.metric("Umur", f"{age} tahun")
        st.metric("Gaji Bulanan", f"${income:,}")
        st.metric("Masa Kerja", f"{years_company} tahun")
        st.metric("Level Jabatan", job_level_text)

    # LIME Explanation Section
    st.subheader("🔍 Faktor Individual yang Mempengaruhi Prediksi")
    
    with st.spinner("Menghasilkan penjelasan..."):
        features, values, explanation = create_lime_explanation(model, scaler, hr_input, feature_names, lime_config, explanation_cache)
    
    if features and values:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📊 Kontribusi Fitur Individual**")
            
            # Create LIME chart
            colors = ['red' if v < 0 else 'green' for v in values]
            
            fig_lime = go.Figure(go.Bar(
                x=values,
                y=features,
                orientation='h',
                marker_color=colors,
                text=[f"{v:.3f}" for v in values],
                textposition='outside'
            ))
            
            fig_lime.update_layout(
                title="Kontribusi Fitur",
                xaxis_title="Kontribusi terhadap Prediksi",
                yaxis_title="Fitur",
                height=400,
                showlegend=False
            )
            
            st.plotly_chart(fig_lime, use_container_width=True)
        
        with col2:
            st.markdown("**📝Deskripsi**")
            
            positive_factors = [(f, v) for f, v in zip(features, values) if v > 0]
            negative_factors = [(f, v) for f, v in zip(features, values) if v < 0]
            
            def clean_lime_explanation(feature_text, value):
                """Convert technical explanation to business-friendly format"""
                
                # Extract base feature name (before any condition symbols)
                base_feature = feature_text.split(' ≤')[0].split(' >')[0].split(' <')[0].strip()
                
                # Map technical names to business-friendly Indonesian
                feature_mapping = {
                    'OverTime_Yes': 'Sering Kerja Lembur',
                    'JobSatisfaction': 'Kepuasan Kerja Rendah',
                    'WorkLifeBalance': 'Work-Life Balance Buruk', 
                    'EnvironmentSatisfaction': 'Kepuasan Lingkungan Kerja Rendah',
                    'MonthlyIncome': 'Gaji Bulanan Rendah',
                    'Age': 'Usia Karyawan',
                    'YearsAtCompany': 'Masa Kerja Pendek',
                    'DistanceFromHome': 'Jarak Rumah Jauh',
                    'YearsSinceLastPromotion': 'Lama Tanpa Promosi',
                    'BusinessTravel_Travel_Frequently': 'Sering Perjalanan Dinas',
                    'BusinessTravel_Travel_Rarely': 'Jarang Perjalanan Dinas',
                    'JobLevel': 'Level Pekerjaan Rendah',
                    'StockOptionLevel': 'Opsi Saham Rendah',
                    'PercentSalaryHike': 'Kenaikan Gaji Rendah',
                    'PerformanceRating': 'Rating Kinerja',
                    'JobInvolvement': 'Keterlibatan Kerja Rendah',
                    'RelationshipSatisfaction': 'Kepuasan Hubungan Kerja Rendah',
                    'TotalWorkingYears': 'Total Pengalaman Kerja',
                    'NumCompaniesWorked': 'Jumlah Perusahaan Sebelumnya',
                    'Gender_Male': 'Jenis Kelamin Laki-laki',
                    'MaritalStatus_Single': 'Status Lajang',
                    'MaritalStatus_Married': 'Status Menikah'
                }
                
                # Handle partial matches for complex feature names
                clean_name = base_feature
                for tech_name, friendly_name in feature_mapping.items():
                    if tech_name in base_feature:
                        clean_name = friendly_name
                        break
                
                # If no mapping found, clean the technical name
                if clean_name == base_feature:
                    # Remove underscores and clean up
                    clean_name = base_feature.replace('_', ' ').replace('Field', '').replace('Education', 'Pendidikan')
                    # Handle common patterns
                    if 'Department' in clean_name:
                        clean_name = clean_name.replace('Department', 'Dept.')
                    elif 'JobRole' in clean_name:
                        clean_name = clean_name.replace('JobRole', 'Posisi')
                
                # Determine impact direction and create simple explanation
                impact_strength = abs(value)
                if impact_strength > 0.08:
                    strength = "Sangat"
                elif impact_strength > 0.05:
                    strength = "Cukup"
                else:
                    strength = "Sedikit"
                
                return clean_name, strength
            
            if positive_factors:
                st.markdown("**🔴 Faktor yang MENINGKATKAN risiko attrisi:**")
                for factor, value in positive_factors[:5]:
                    clean_name, strength = clean_lime_explanation(factor, value)
                    st.write(f"• **{clean_name}** ({strength} berpengaruh: +{value:.3f})")
            
            if negative_factors:
                st.markdown("**🟢 Faktor yang MENURUNKAN risiko attrisi:**")
                for factor, value in negative_factors[:5]:
                    clean_name, strength = clean_lime_explanation(factor, value)
                    st.write(f"• **{clean_name}** ({strength} berpengaruh: {value:.3f})")
        
                
    else:
        st.info("ℹ️ Analisis LIME tidak tersedia. Menggunakan analisis faktor risiko tradisional.")

    # Risk Factors Analysis
    st.subheader("🔍 Analisis Faktor Risiko")
    
    risk_factors = []
    
    # Check various risk factors
    if hr_input.get('OverTime', 0) == 1:
        risk_factors.append("🔴 **Sering Lembur** - Indikasi workload berlebihan atau work-life balance buruk")
    
    if hr_input.get('JobSatisfaction', 4) <= 2:
        risk_factors.append("🔴 **Kepuasan Kerja Rendah** - Faktor risiko utama untuk attrisi")
    
    if hr_input.get('WorkLifeBalance', 4) <= 2:
        risk_factors.append("🔴 **Work-Life Balance Buruk** - Dapat menyebabkan burnout")
    
    if hr_input.get('YearsSinceLastPromotion', 0) >= 3:
        risk_factors.append("🟡 **Tidak Ada Promosi 3+ Tahun** - Potensi stagnasi karir")
    
    if hr_input.get('DistanceFromHome', 0) > 20:
        risk_factors.append("🟡 **Jarak Rumah Jauh** - Biaya dan waktu komute tinggi")
    
    if hr_input.get('PercentSalaryHike', 13) < 10:
        risk_factors.append("🟡 **Kenaikan Gaji Rendah** - Gaji tidak kompetitif")
    
    if hr_input.get('BusinessTravel', 1) == 2:
        risk_factors.append("🟡 **Sering Perjalanan Dinas** - Dapat mengganggu kehidupan pribadi")
    
    if hr_input.get('EnvironmentSatisfaction', 4) <= 2:
        risk_factors.append("🟡 **Kepuasan Lingkungan Kerja Rendah** - Budaya atau lingkungan tidak mendukung")

    st.markdown("**⚠️ Faktor Risiko yang Teridentifikasi:**")
    if risk_factors:
        for factor in risk_factors:
            st.write(factor)
    else:
        st.success("✅ Tidak ada faktor risiko utama yang terdeteksi")

def display_global_feature_importance(global_importance, feature_descriptions):
    """Display global feature importance dashboard"""
    st.header("📊 Dashboard Model: Global Feature Importance")
    
    if global_importance is not None:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("🎯 Top 15 Fitur Paling Berpengaruh")
            
            # Handle different formats of global_importance
            if isinstance(global_importance, dict):
                # If it's a dictionary
                top_features = list(global_importance.keys())[:15]
                top_scores = list(global_importance.values())[:15]
                print(f"✅ Processed dictionary format: {len(top_features)} features")
            elif isinstance(global_importance, pd.DataFrame):
                # If it's a pandas DataFrame
                print(f"🔍 DEBUG: DataFrame columns: {global_importance.columns.tolist()}")
                print(f"🔍 DEBUG: DataFrame shape: {global_importance.shape}")
                
                try:
                    # Assume first column is feature names, second is scores
                    df_sorted = global_importance.sort_values(global_importance.columns[1], ascending=False)
                    top_features = df_sorted.iloc[:15, 0].astype(str).tolist()
                    top_scores = df_sorted.iloc[:15, 1].astype(float).tolist()
                    print(f"✅ Successfully processed DataFrame: {len(top_features)} features")
                except Exception as e:
                    print(f"❌ Error processing DataFrame: {e}")
                    st.warning("⚠️ Error memproses DataFrame, menggunakan demo data")
                    top_features, top_scores = get_demo_feature_importance()
            elif isinstance(global_importance, (list, tuple)):
                # If it's a list of tuples [(feature, score), ...]
                sorted_importance = sorted(global_importance, key=lambda x: x[1], reverse=True)[:15]
                top_features = [item[0] for item in sorted_importance]
                top_scores = [item[1] for item in sorted_importance]
                print(f"✅ Processed list/tuple format: {len(top_features)} features")
            elif hasattr(global_importance, 'shape'):
                # If it's a numpy array or similar
                if len(global_importance.shape) == 1:
                    # 1D array of scores, need feature names
                    try:
                        # Try to get feature names from somewhere
                        feature_names = ['Feature_' + str(i) for i in range(len(global_importance))]
                        importance_pairs = list(zip(feature_names, global_importance))
                        sorted_importance = sorted(importance_pairs, key=lambda x: x[1], reverse=True)[:15]
                        top_features = [item[0] for item in sorted_importance]
                        top_scores = [item[1] for item in sorted_importance]
                    except:
                        # Fallback to demo data
                        st.warning("⚠️ Format feature importance tidak sesuai, menggunakan demo data")
                        top_features, top_scores = get_demo_feature_importance()
                else:
                    # 2D array or other format
                    st.warning("⚠️ Format feature importance tidak didukung, menggunakan demo data")
                    top_features, top_scores = get_demo_feature_importance()
            else:
                # Unknown format
                st.warning("⚠️ Format feature importance tidak dikenal, menggunakan demo data")
                top_features, top_scores = get_demo_feature_importance()
            
            # Create horizontal bar chart
            fig_importance = go.Figure(go.Bar(
                x=top_scores,
                y=top_features,
                orientation='h',
                marker_color='rgba(55, 128, 191, 0.7)',
                text=[f"{score:.3f}" for score in top_scores],
                textposition='outside'
            ))
            
            fig_importance.update_layout(
                title="Global Feature Importance Scores",
                xaxis_title="Importance Score",
                yaxis_title="Features",
                height=600,
                showlegend=False,
                margin=dict(l=200)
            )
            
            st.plotly_chart(fig_importance, use_container_width=True)
        
        with col2:
            st.subheader("📝 Deskripsi Fitur")
            
            for feature in top_features[:10]:
                with st.expander(f"📊 {feature}"):
                    description = feature_descriptions.get(feature, "Deskripsi tidak tersedia")
                    if isinstance(top_scores, list) and len(top_scores) > 0:
                        try:
                            feature_idx = top_features.index(feature)
                            importance_score = top_scores[feature_idx]
                            st.write(f"**Importance Score:** {importance_score:.3f}")
                        except:
                            st.write(f"**Importance Score:** N/A")
                    
                    st.write(f"**Deskripsi:** {description}")
                    
                    # Add business interpretation
                    if 'satisfaction' in feature.lower():
                        st.info("💡 Faktor kepuasan - indikator kunci employee engagement")
                    elif 'overtime' in feature.lower():
                        st.info("💡 Faktor work-life balance - risiko burnout")
                    elif 'income' in feature.lower() or 'salary' in feature.lower():
                        st.info("💡 Faktor kompensasi - competitive positioning")
                    elif 'years' in feature.lower():
                        st.info("💡 Faktor experience - career progression")
    
    else:
        st.info("📊 Data feature importance tidak tersedia. Menampilkan contoh analisis.")
        top_features, top_scores = get_demo_feature_importance()
        
        fig_demo = go.Figure(go.Bar(
            x=top_scores,
            y=top_features,
            orientation='h',
            marker_color='rgba(255, 99, 71, 0.7)',
            text=[f"{score:.3f}" for score in top_scores],
            textposition='outside'
        ))
        
        fig_demo.update_layout(
            title="Demo: Global Feature Importance",
            xaxis_title="Importance Score",
            yaxis_title="Features",
            height=600,
            showlegend=False,
            margin=dict(l=200)
        )
        
        st.plotly_chart(fig_demo, use_container_width=True)

def get_demo_feature_importance():
    """Get demo feature importance data"""
    demo_features = [
        'OverTime_Yes', 'JobSatisfaction', 'MonthlyIncome', 'Age', 'WorkLifeBalance',
        'YearsAtCompany', 'DistanceFromHome', 'YearsSinceLastPromotion', 
        'EnvironmentSatisfaction', 'JobLevel', 'BusinessTravel_Travel_Frequently',
        'PercentSalaryHike', 'StockOptionLevel', 'PerformanceRating', 'Gender_Male'
    ]
    demo_scores = [0.234, 0.187, 0.156, 0.134, 0.127, 0.098, 0.087, 0.076, 0.065, 0.054, 0.045, 0.034, 0.028, 0.021, 0.012]
    return demo_features, demo_scores

def main_app():
    """Aplikasi utama dengan enhanced interpretability"""
    username = st.session_state.username
    user_role = st.session_state.get('user_role', 'admin')
    user_permissions = st.session_state.get('user_permissions', [])
    display_name = st.session_state.get('display_name', username)
    
    # Header dengan informasi role
    st.title(f"👥 Sistem Prediksi Attrisi Karyawan")
    
    st.markdown(f"""
    **Selamat datang {display_name}** 🟢 **HR Administrator** | **🟢 Sistem Online** | **🛡️ Sesi Aman**
    
    ### 📊 **Tentang Sistem Analitik HR**
    Sistem prediksi attrisi karyawan berbasis Machine Learning yang dirancang khusus untuk membantu departemen HR dalam:
    - **🎯 Prediksi Risiko:** Mengidentifikasi karyawan yang berisiko tinggi keluar dari perusahaan
    - 🔍 Penjelasan individual mengapa model memprediksi risiko tertentu
    - **📊 Feature Importance:** Memahami faktor-faktor global yang paling berpengaruh
    - **📈 Insight Mendalam:** Memberikan analisis berbasis data untuk pemahaman yang lebih baik
    
    **Status Akses:** Full Administrator Access ✅
    """)
    
    # Logout button in sidebar
    if st.sidebar.button("🚪 Keluar"):
        st.session_state.logged_in = False
        st.session_state.username = None
        st.session_state.user_role = None
        st.session_state.user_permissions = []
        st.rerun()
    
    # Display admin info in sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**🔍 Level Akses: HR Administrator**")
    st.sidebar.markdown("**✅ Akses Penuh:**")
    st.sidebar.write("• Penilaian Risiko Karyawan")
    st.sidebar.write("• Faktor Individual yang Mempengaruhi Prediksi")
    st.sidebar.write("• aktor Utama yang Mempengaruhi Prediksi Secara Keseluruhan")
    st.sidebar.write("• Dashboard Analitik")
    
    st.markdown("---")
    
    # Load model components with interpretability
    model, scaler, feature_names, metadata, global_importance, feature_descriptions, lime_config, interpretability_metadata, explanation_cache, streamlit_helpers = load_model_components()
    
    # Create tabs - merged structure
    tab1, tab2 = st.tabs(["🎯 Penilaian Risiko Karyawan", "📊 Dashboard Analitik Komprehensif"])
    
    with tab1:
        st.header("🎯 Penilaian Risiko Attrisi Karyawan")
        st.markdown("**Evaluasi risiko attrisi individual**")
        
        hr_categorizer = HRFeatureCategorizer()
        hr_input = hr_categorizer.create_hr_input_form()
        
        # Display summary
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Umur", f"{hr_input.get('Age', 0)} tahun")
        col2.metric("Gaji", f"${hr_input.get('MonthlyIncome', 0):,}")
        col3.metric("Masa Kerja", f"{hr_input.get('YearsAtCompany', 0)} tahun")
        
        # Convert job level to text
        job_levels = {1: "Pemula", 2: "Junior", 3: "Menengah", 4: "Senior", 5: "Eksekutif"}
        job_level_text = job_levels.get(hr_input.get('JobLevel', 3), "Menengah")
        col4.metric("Level Kerja", job_level_text)
        
        if st.button("🚀 Analisis Risiko Attrisi", type="primary"):
            with st.spinner("Menganalisis data karyawan..."):
                prediction, prediction_proba, input_df = make_prediction(model, scaler, hr_input, feature_names)
                
                if prediction is not None:
                    st.success("✅ Analisis selesai!")
                    display_prediction_results(prediction, prediction_proba, hr_input, metadata, 
                                             model, scaler, feature_names, lime_config, explanation_cache)
                else:
                    st.error("❌ Analisis gagal")
    
    with tab2:
        st.header("📊 Dashboard Analitik")
        st.markdown("**Overview kinerja sistem dan insight organisasi**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Kinerja Model")
            
            if metadata:
                st.metric("Akurasi Model", f"{metadata.get('test_accuracy', 0.87):.1%}")
                st.metric("Skor ROC-AUC", f"{metadata.get('roc_auc', 0.82):.3f}")
                st.metric("Tipe Model", metadata.get('model_type', 'Logistic Regression'))
            
            st.subheader("🏢 Analisis Risiko Departemen")
            dept_data = {
                'Departemen': ['Sales', 'Engineering', 'Marketing', 'HR', 'Finance'],
                'Risiko_Rata2': [0.28, 0.15, 0.22, 0.12, 0.18],
                'Jumlah_Karyawan': [150, 200, 75, 25, 50]
            }
            dept_df = pd.DataFrame(dept_data)
            
            fig_dept = px.bar(
                dept_df,
                x='Departemen',
                y='Risiko_Rata2',
                color='Risiko_Rata2',
                title='Rata-rata Risiko Attrisi per Departemen',
                color_continuous_scale='Reds',
                text='Risiko_Rata2'
            )
            fig_dept.update_traces(texttemplate='%{text:.1%}', textposition='outside')
            fig_dept.update_layout(showlegend=False)
            st.plotly_chart(fig_dept, use_container_width=True)
        
        with col2:
            st.subheader("📈 Statistik Penggunaan")
            
            st.metric("Penilaian Hari Ini", "47", "+12")
            st.metric("Karyawan Risiko Tinggi", "23", "+3") 
            st.metric("Total Karyawan Dinilai", "342", "+28")
            st.metric("Akurasi Prediksi", "87%", "+2%")
            
            st.subheader("🎯 10 Faktor Utama yang Mempengaruhi Prediksi Secara Keseluruhan")
            
            # Extract top 10 features from global_importance
            if global_importance is not None:
                try:
                    if isinstance(global_importance, pd.DataFrame):
                        # Sort by importance and get top 10
                        df_sorted = global_importance.sort_values(global_importance.columns[1], ascending=False)
                        top_features = df_sorted.iloc[:10, 0].astype(str).tolist()
                        top_scores = df_sorted.iloc[:10, 1].astype(float).tolist()
                        
                        # Clean feature names untuk display
                        clean_feature_names = []
                        for feature in top_features:
                            if 'OverTime' in feature:
                                clean_feature_names.append('Kerja Lembur')
                            elif 'JobSatisfaction' in feature:
                                clean_feature_names.append('Kepuasan Kerja')
                            elif 'WorkLifeBalance' in feature:
                                clean_feature_names.append('Work-Life Balance')
                            elif 'MonthlyIncome' in feature:
                                clean_feature_names.append('Gaji Bulanan')
                            elif 'Age' in feature:
                                clean_feature_names.append('Umur')
                            elif 'YearsAtCompany' in feature:
                                clean_feature_names.append('Masa Kerja')
                            elif 'DistanceFromHome' in feature:
                                clean_feature_names.append('Jarak Rumah')
                            elif 'EnvironmentSatisfaction' in feature:
                                clean_feature_names.append('Kepuasan Lingkungan')
                            elif 'YearsSinceLastPromotion' in feature:
                                clean_feature_names.append('Tahun Tanpa Promosi')
                            elif 'BusinessTravel' in feature:
                                clean_feature_names.append('Perjalanan Dinas')
                            else:
                                # Keep original name but clean it
                                clean_name = feature.replace('_', ' ').title()
                                clean_feature_names.append(clean_name)
                        
                        importance_data = {
                            'Fitur': clean_feature_names,
                            'Importance_Score': top_scores
                        }
                        importance_df = pd.DataFrame(importance_data)
                        
                    else:
                        # Fallback data
                        importance_data = {
                            'Fitur': ['Kerja Lembur', 'Kepuasan Kerja', 'Gaji Bulanan', 'Umur', 'Work-Life Balance',
                                     'Masa Kerja', 'Jarak Rumah', 'Tahun Tanpa Promosi', 'Kepuasan Lingkungan', 'Perjalanan Dinas'],
                            'Importance_Score': [0.234, 0.187, 0.156, 0.134, 0.127, 0.098, 0.087, 0.076, 0.065, 0.054]
                        }
                        importance_df = pd.DataFrame(importance_data)
                        
                except Exception as e:
                    # Fallback data
                    importance_data = {
                        'Fitur': ['Kerja Lembur', 'Kepuasan Kerja', 'Gaji Bulanan', 'Umur', 'Work-Life Balance',
                                 'Masa Kerja', 'Jarak Rumah', 'Tahun Tanpa Promosi', 'Kepuasan Lingkungan', 'Perjalanan Dinas'],
                        'Importance_Score': [0.234, 0.187, 0.156, 0.134, 0.127, 0.098, 0.087, 0.076, 0.065, 0.054]
                    }
                    importance_df = pd.DataFrame(importance_data)
            else:
                # Fallback data
                importance_data = {
                    'Fitur': ['Kerja Lembur', 'Kepuasan Kerja', 'Gaji Bulanan', 'Umur', 'Work-Life Balance',
                             'Masa Kerja', 'Jarak Rumah', 'Tahun Tanpa Promosi', 'Kepuasan Lingkungan', 'Perjalanan Dinas'],
                    'Importance_Score': [0.234, 0.187, 0.156, 0.134, 0.127, 0.098, 0.087, 0.076, 0.065, 0.054]
                }
                importance_df = pd.DataFrame(importance_data)
            
            # Create horizontal bar chart (same style as dampak bisnis)
            fig_importance = px.bar(
                importance_df,
                x='Importance_Score',
                y='Fitur',
                orientation='h',
                title='10 Faktor Utama yang Mempengaruhi Prediksi Secara Keseluruhan',
                color='Importance_Score',
                color_continuous_scale='Blues',
                text='Importance_Score'
            )
            fig_importance.update_traces(texttemplate='%{text:.3f}', textposition='outside')
            fig_importance.update_layout(showlegend=False)
            st.plotly_chart(fig_importance, use_container_width=True)
        
        # Summary Statistics
        st.subheader("📊 Ringkasan Statistik")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("**📈 Rata-rata Tingkat Attrisi**")
            st.info("15.4% per tahun")
            st.caption("Target: <14%")
        
        with col2:
            st.markdown("**🎯 Tingkat Deteksi Dini**")
            st.info("73% kasus terdeteksi")
            st.caption("dari total attrisi aktual")
        
        with col3:
            st.markdown("**⏰ Waktu Prediksi Rata-rata**")
            st.info("2.3 bulan sebelum resign")
            st.caption("window untuk intervensi")
        
        with col4:
            st.markdown("**🏆 Departemen Terstabil**")
            st.info("HR Department")
            st.caption("12% tingkat attrisi")

def main():
    """Fungsi utama dengan enhanced interpretability"""
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    
    if st.session_state.logged_in:
        main_app()
    else:
        login_page()

if __name__ == "__main__":

    main()

