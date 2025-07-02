import streamlit as st
import pandas as pd
import os
import warnings
from sklearn.exceptions import InconsistentVersionWarning
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


# Import các module con
try:
    from sentiment_analysis_page import sentiment_analysis_app
except ImportError as e:
    st.error(f"Không thể import sentiment_analysis_page: {e}")
    sentiment_analysis_app = None

try:
    from information_clustering_page import information_clustering_app
except ImportError as e:
    st.error(f"Không thể import information_clustering_page: {e}")
    information_clustering_app = None

# Cấu hình trang
st.set_page_config(
    page_title="Sentiment & Clustering Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS để tùy chỉnh giao diện với theme tối
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* Main app background */
    .stApp {
        background-color: #1a1a1a;
        color: #ffffff;
    }
    
    /* Main container styling */
    .main {
        font-family: 'Inter', sans-serif;
        background-color: #1a1a1a;
        color: #ffffff;
    }
    
    /* Header styling */
    .header-container {
        background: linear-gradient(135deg, #4c1d95 0%, #5b21b6 50%, #7c3aed 100%);
        padding: 2rem 1rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
        box-shadow: 0 8px 32px rgba(139, 92, 246, 0.3);
        border: 1px solid rgba(139, 92, 246, 0.2);
    }
    
    .header-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
    }
    
    .header-subtitle {
        font-size: 1.2rem;
        opacity: 0.9;
        font-weight: 400;
    }
    
    /* Breadcrumb styling */
    .breadcrumb {
        background: linear-gradient(90deg, #2d3748 0%, #374151 100%);
        padding: 1rem 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #8b5cf6;
        margin: 1rem 0;
        font-weight: 500;
        color: #e5e7eb;
        box-shadow: 0 2px 10px rgba(0,0,0,0.3);
    }
    
    /* Card styling */
    .info-card {
        background: linear-gradient(135deg, #2d3748 0%, #374151 100%);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        border: 1px solid #4a5568;
        margin: 1rem 0;
        color: #e5e7eb;
    }
    
    .info-card h3, .info-card h4 {
        color: #f7fafc;
        margin-bottom: 1rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #2d3748 0%, #374151 100%);
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 2px 10px rgba(0,0,0,0.3);
        border: 1px solid #4a5568;
        color: #e5e7eb;
    }
    
    .metric-card h2, .metric-card h3 {
        color: #f7fafc;
    }
    
    /* Footer styling */
    .footer-container {
        background: linear-gradient(135deg, #2d3748 0%, #374151 100%);
        border-radius: 15px;
        margin-top: 2rem;
        padding: 2rem;
        text-align: center;
        border: 1px solid #4a5568;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
    
    .footer-main-title {
        color: #e5e7eb;
        font-size: 1.4rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .footer-subtitle {
        color: #d1d5db;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }
    
    .footer-credits {
        color: #9ca3af;
        font-size: 0.95rem;
        margin-bottom: 1rem;
    }
    
    .footer-credits strong {
        color: #e5e7eb;
    }
    
    .footer-note {
        color: #9ca3af;
        font-size: 0.85rem;
        font-style: italic;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #7c3aed 0%, #8b5cf6 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(139, 92, 246, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(139, 92, 246, 0.4);
        background: linear-gradient(135deg, #8b5cf6 0%, #a855f7 100%);
    }
    
    /* Sidebar styling */
    .css-1d391kg, .css-1cypcdb {
        background-color: #2d3748 !important;
    }
    
    /* Selectbox styling */
    .stSelectbox > div > div {
        background-color: #374151;
        border-radius: 8px;
        border: 2px solid #4a5568;
        color: #e5e7eb;
    }
    
    /* Text input styling */
    .stTextInput > div > div > input, .stTextArea > div > div > textarea {
        background-color: #374151;
        border: 2px solid #4a5568;
        color: #e5e7eb;
        border-radius: 8px;
    }
    
    .stTextInput > div > div > input:focus, .stTextArea > div > div > textarea:focus {
        border-color: #8b5cf6;
        box-shadow: 0 0 0 2px rgba(139, 92, 246, 0.2);
    }
    
    /* Slider styling */
    .stSlider > div > div > div {
        background-color: #374151;
    }
    
    /* Success/Info boxes */
    .success-box {
        background: linear-gradient(135deg, #064e3b 0%, #065f46 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #10b981;
        margin: 1rem 0;
        color: #d1fae5;
        box-shadow: 0 2px 10px rgba(0,0,0,0.3);
    }
    
    .info-box {
        background: linear-gradient(135deg, #1e3a8a 0%, #1d4ed8 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #3b82f6;
        margin: 1rem 0;
        color: #dbeafe;
        box-shadow: 0 2px 10px rgba(0,0,0,0.3);
    }
    
    /* Table styling */
    .stDataFrame {
        background-color: #374151;
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 2px 10px rgba(0,0,0,0.3);
    }
    
    /* Section headers */
    .section-header {
        color: #f7fafc;
        font-weight: 600;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #4a5568;
    }
    
    /* Prediction result */
    .prediction-result {
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        font-size: 1.2rem;
        font-weight: 600;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #2d3748;
        border-radius: 10px;
        padding: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        color: #9ca3af;
        border-radius: 8px;
        padding: 0.5rem 1rem;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #8b5cf6;
        color: white;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #374151;
        color: #e5e7eb;
        border-radius: 8px;
    }
    
    /* Radio button styling */
    .stRadio > div {
        background-color: #374151;
        border-radius: 8px;
        padding: 1rem;
    }
    
    /* Metric styling */
    [data-testid="metric-container"] {
        background-color: #374151;
        border: 1px solid #4a5568;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.2);
    }
    
    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #2d3748;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #4a5568;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #6b7280;
    }
</style>
""", unsafe_allow_html=True)

# Đọc dữ liệu với error handling
@st.cache_data
def load_data():
    try:
        if os.path.exists('processed_reviews.xlsx'):
            return pd.read_excel('processed_reviews.xlsx')
        else:
            st.error("Không tìm thấy file 'processed_reviews.xlsx'")
            return None
    except Exception as e:
        st.error(f"Lỗi khi đọc dữ liệu: {e}")
        return None

# Load dữ liệu
if 'reviews' not in st.session_state:
    st.session_state.reviews = load_data()

df_reviews = st.session_state.reviews

# Kiểm tra nếu dữ liệu không load được
if df_reviews is None:
    st.stop()

###### GUI ######
# Header với thiết kế đẹp (giảm icon)
st.markdown("""
<div class="header-container">
    <div class="header-title">Sentiment & Clustering Analysis</div>
    <div class="header-subtitle">Phân tích cảm xúc và phân cụm thông tin từ đánh giá ITviec</div>
</div>
""", unsafe_allow_html=True)

# Sidebar (giảm icon)
st.sidebar.markdown("### MENU")

# Menu cấp 1
choice_lv1 = st.sidebar.selectbox(
    'Menu chính', 
    ['Tổng quan', 'Sentiment Analysis', 'Information Clustering'],
    help="Chọn phần bạn muốn khám phá"
)

# Menu cấp 2
menu_lv2 = {
    'Tổng quan': ['Giới thiệu'],
    'Sentiment Analysis': ["Business Objective", "Build Project", "New Prediction"],
    'Information Clustering': ["Business Objective", "Build Project", "New Prediction"]
}

choice_lv2 = st.sidebar.selectbox(
    f'{choice_lv1}', 
    menu_lv2.get(choice_lv1, []),
    help=f"Chọn mục con trong {choice_lv1}"
)

# Breadcrumb (giảm icon)
st.markdown(f"""
<div class="breadcrumb">
    {choice_lv1} → {choice_lv2}
</div>
""", unsafe_allow_html=True)

# === Nội dung hiển thị theo từng mục ===
if choice_lv1 == 'Tổng quan':
    if choice_lv2 == 'Giới thiệu':
        st.markdown('<h1 class="section-header">Chào mừng đến với ứng dụng phân tích</h1>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            <div class="info-card">
                <h3>Sentiment Analysis</h3>
                <p><strong>Sentiment Analysis</strong> là quá trình sử dụng xử lý ngôn ngữ tự nhiên và học máy để phân tích cảm xúc trong các đánh giá, phản hồi từ người dùng (tích cực, tiêu cực, trung lập).</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="info-card">
                <h3>Information Clustering</h3>
                <p><strong>Information Clustering</strong> giúp phân nhóm các đánh giá để doanh nghiệp hiểu rõ họ thuộc nhóm nào → từ đó cải thiện và phát triển tốt hơn.</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="info-card">
                <h3>Nguồn dữ liệu</h3>
                <p>Dữ liệu từ <strong>ITviec.com</strong></p>
                <p>Reviews từ ứng viên và nhân viên</p>
            </div>
            """, unsafe_allow_html=True)
            
        # Thông tin tổng quan về dữ liệu
        st.markdown('<h2 class="section-header">Thông tin tổng quan về dữ liệu</h2>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Tổng số review", f"{len(df_reviews):,}")
        with col2:
            st.metric("Số công ty", f"{df_reviews['Company Name'].nunique():,}")
        with col3:
            if 'reviews_text' in df_reviews.columns:
                avg_length = df_reviews['reviews_text'].str.len().mean()
                st.metric("Độ dài TB", f"{avg_length:.0f} ký tự")
            else:
                st.metric("Độ dài TB", "N/A")
        with col4:
            if 'reviews_text' in df_reviews.columns:
                avg_words = df_reviews['reviews_text'].str.split().str.len().mean()
                st.metric("Từ TB/review", f"{avg_words:.0f} từ")
            else:
                st.metric("Từ TB/review", "N/A")
        
        # Hiển thị sample data
        st.markdown('<h3 class="section-header">Mẫu dữ liệu</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Top 5 đánh giá đầu tiên:**")
            display_cols = ['Company Name', 'reviews_text'] if 'reviews_text' in df_reviews.columns else df_reviews.columns[:2]
            st.dataframe(df_reviews[display_cols].head(5), use_container_width=True)
        
        with col2:
            st.markdown("**5 đánh giá cuối cùng:**")
            st.dataframe(df_reviews[display_cols].tail(5), use_container_width=True)

elif choice_lv1 == 'Sentiment Analysis':
    # Gọi ứng dụng sentiment analysis
    if sentiment_analysis_app is not None:
        sentiment_analysis_app(choice_lv2, df_reviews)
    else:
        st.error("Không thể load module Sentiment Analysis")
        st.info("Vui lòng kiểm tra file sentiment_analysis_page.py")

elif choice_lv1 == 'Information Clustering':
    # Gọi ứng dụng information clustering
    if information_clustering_app is not None:
        information_clustering_app(choice_lv2, df_reviews)
    else:
        st.error("Không thể load module Information Clustering")
        st.info("Vui lòng kiểm tra file information_clustering_page.py")

# Footer với thông tin tác giả
st.markdown("---")

# Footer căn giữa
st.markdown("""
<style>
.footer-container {
    width: 100%;
    margin: 40px auto;
    padding: 25px;
    border-radius: 12px;
    background-color: #2c3e50;
    text-align: center;
    font-family: 'Segoe UI', sans-serif;
    color: #ecf0f1;
    box-shadow: 0 4px 12px rgba(0,0,0,0.2);
}
.footer-container h4 {
    font-size: 18px;
    font-weight: 600;
}
.footer-container .title {
    font-size: 20px;
    font-weight: bold;
    margin-bottom: 10px;
    color: #f1c40f;
}
.footer-container a {
    color: #1abc9c;
    text-decoration: none;
}
.footer-container a:hover {
    text-decoration: underline;
}
.footer-container hr {
    margin: 18px auto;
    width: 60%;
    border: 0.5px solid #7f8c8d;
}
.footer-container p {
    margin: 6px 0;
    font-size: 15px;
}
</style>

<div class="footer-container">
    <p class="title">🎓 Đồ án tốt nghiệp – Data Science & Machine Learning</p>
    <h4>Phát triển bởi</h4>
    <p>• <strong>Trần Hoàng Hôn</strong> – <a href="mailto:hoanghonhutech@gmail.com">hoanghonhutech@gmail.com</a></p>
    <p>• <strong>Trương Mai</strong> – <a href="mailto:trgmai98.dev@gmail.com">trgmai98.dev@gmail.com</a></p>
    <hr>
    <p><em>Made with ❤️ using <strong>Streamlit</strong> & <strong>Machine Learning</strong></em></p>
</div>
""", unsafe_allow_html=True)
