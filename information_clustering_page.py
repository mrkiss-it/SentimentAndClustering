
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse import hstack, csr_matrix
import joblib

# Import sentiment analysis functions với error handling
try:
    from sentiment.sentiment_analysis import *
except ImportError as e:
    st.error(f"❌ Không thể import sentiment analysis module: {e}")
except Exception as e:
    st.error(f"❌ Lỗi khi import: {e}")

def check_wordcloud(data, col_name):
    """Tạo WordCloud từ dữ liệu text"""
    text = " ".join(data)  # Gộp danh sách thành chuỗi
    wc = WordCloud(width=800, height=400, background_color='white').generate(text)

    # Tạo figure và vẽ WordCloud
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis("off")
    ax.set_title("WordCloud của " + col_name, fontsize=16, fontweight='bold', pad=20)

    return fig

def information_clustering_app(choice_lv2_clean, df_reviews):
    """Main function cho Information Clustering app"""
    
    # Load clustering models theo pattern từ file gốc
    scaler = MinMaxScaler()
    cluster_names = ['EXCELLENT', 'AVERAGE', 'PROBLEMATIC']
    
    try:
        clustering_vectorizer = joblib.load("clustering/tfidf_vectorizer.pkl")
        clustering_model = joblib.load("clustering/best_prediction_model.pkl")
        models_loaded = True
    except FileNotFoundError as e:
        st.error(f"❌ Không tìm thấy file model: {e}")
        st.info("💡 Vui lòng kiểm tra các file sau có tồn tại:")
        st.info("- clustering/tfidf_vectorizer.pkl")
        st.info("- clustering/best_prediction_model.pkl")
        clustering_vectorizer = None
        clustering_model = None
        models_loaded = False
    except Exception as e:
        st.error(f"❌ Lỗi khi load models: {e}")
        clustering_vectorizer = None
        clustering_model = None
        models_loaded = False

    if choice_lv2_clean == "Business Objective":
        st.markdown('<h1 class="section-header">📌 Mục tiêu phân cụm thông tin</h1>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="info-card">
                <h3>🎯 Yêu cầu</h3>
                <p><strong>Phân nhóm review</strong> theo nội dung và cảm nhận để hiểu rõ đặc điểm của từng nhóm công ty.</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="info-card">
                <h3>🎯 Mục tiêu</h3>
                <p>Mỗi công ty biết mình thuộc cụm nào → đưa ra <strong>chiến lược cải thiện</strong> phù hợp.</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Cluster với màu tối đẹp hơn
        st.markdown("### 🏷️ Các nhóm phân cụm")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="metric-card" style="background: linear-gradient(135deg, #065f46 0%, #047857 100%); border-color: #10b981; color: #d1fae5;">
                <h3 style="color: #6ee7b7;">🏆 EXCELLENT</h3>
                <p>Công ty xuất sắc với đánh giá rất tích cực</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card" style="background: linear-gradient(135deg, #92400e 0%, #b45309 100%); border-color: #f59e0b; color: #fef3c7;">
                <h3 style="color: #fcd34d;">⚖️ AVERAGE</h3>
                <p>Công ty trung bình, cần cải thiện một số mặt</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-card" style="background: linear-gradient(135deg, #991b1b 0%, #b91c1c 100%); border-color: #dc2626; color: #fecaca;">
                <h3 style="color: #fca5a5;">⚠️ PROBLEMATIC</h3>
                <p>Công ty có nhiều vấn đề cần giải quyết</p>
            </div>
            """, unsafe_allow_html=True)

    elif choice_lv2_clean == "Build Project":
        st.markdown('<h1 class="section-header">🏗️ Xây dựng mô hình phân cụm thông tin</h1>', unsafe_allow_html=True)

        # Tabs giống sentiment_analysis_page
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Dữ liệu", "☁️ WordCloud", "🤖 Mô hình", "📈 Kết quả"])

        with tab1:
            st.markdown("### 📋 Dữ liệu mẫu từ review")
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**🔝 Top 3 đánh giá đầu tiên:**")
                st.dataframe(df_reviews[['Company Name', 'reviews_text']].head(3), use_container_width=True)

            with col2:
                st.markdown("**🔚 3 đánh giá cuối cùng:**")
                st.dataframe(df_reviews[['Company Name', 'reviews_text']].tail(3), use_container_width=True)

            st.markdown("### 📝 Thông tin tổng quan dữ liệu")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Tổng số review", len(df_reviews))
            with col2:
                st.metric("Số công ty", df_reviews['Company Name'].nunique())
            with col3:
                st.metric("Số cột dữ liệu", len(df_reviews.columns))

        with tab2:
            st.markdown("### ☁️ Trực quan hóa WordCloud toàn bộ review")

            if 'clean_advance_text2' in df_reviews.columns:
                with st.spinner('Đang tạo WordCloud...'):
                    try:
                        fig_wc = check_wordcloud(df_reviews['clean_advance_text2'].dropna(), 'Reviews')
                        st.pyplot(fig_wc, use_container_width=True)
                    except Exception as e:
                        st.error(f"❌ Không thể tạo WordCloud: {e}")
            else:
                st.warning("⚠️ Không tìm thấy cột 'clean_advance_text2' trong dữ liệu")

            st.markdown("### 🔧 Quá trình tiền xử lý văn bản")
            st.markdown("""
            <div class="info-box">
                <h4>📝 Các bước tiền xử lý:</h4>
                <ul>
                    <li>🧹 Làm sạch văn bản (loại bỏ ký tự đặc biệt, số)</li>
                    <li>✂️ Phân đoạn câu theo ngữ nghĩa</li>
                    <li>🔤 Chuẩn hóa chữ hoa/thường</li>
                    <li>🚫 Loại bỏ stopwords</li>
                    <li>📊 Vector hóa bằng TF-IDF</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        with tab3:
            st.markdown("### 🤖 Thuật toán phân cụm")

            st.markdown("""
            <div class="info-box">
                <h4>🎯 Thuật toán KMeans</h4>
                <p>Được chọn làm mô hình tốt nhất cho bài toán phân cụm review công ty</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("### 📊 So sánh các thuật toán phân cụm")
            st.markdown("""
            | Thuật toán          | Silhouette Score | Ưu điểm                           | Nhược điểm                     |
            |---------------------|------------------|-----------------------------------|--------------------------------|
            | KMeans              | 0.3247           | Nhanh, đơn giản, hiệu quả        | Cần biết trước số cụm          |
            | Hierarchical        | 0.2891           | Không cần biết trước số cụm       | Chậm với dữ liệu lớn           |
            | DBSCAN              | 0.2156           | Tìm cụm bất kỳ hình dạng          | Nhạy cảm với tham số           |
            | Gaussian Mixture    | 0.2634           | Mô hình xác suất, linh hoạt       | Phức tạp, tốn tài nguyên       |
            """)

            st.markdown("### 🔄 Quy trình xây dựng")
            st.code("""
    # Bước 1: Tiền xử lý dữ liệu
    - Làm sạch văn bản
    - Vector hóa TF-IDF cho text
    - Chuẩn hóa MinMaxScaler cho numerical features

    # Bước 2: Kết hợp đặc trưng
    - Kết hợp TF-IDF vector và numerical features
    - Sử dụng scipy.sparse.hstack để tối ưu bộ nhớ

    # Bước 3: Huấn luyện mô hình
    - Khởi tạo KMeans với k=3
    - Fit mô hình trên dữ liệu đã được chuẩn bị
    - Đánh giá chất lượng cụm bằng Silhouette Score

    # Bước 4: Gán nhãn cụm
    - Phân tích đặc điểm từng cụm
    - Gán tên có ý nghĩa: EXCELLENT, AVERAGE, PROBLEMATIC
    """)

        with tab4:
            st.markdown("### 📈 Kết quả phân cụm")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### 📊 Chất lượng cụm")

                if 'cluster_label' in df_reviews.columns:
                    cluster_counts = df_reviews['cluster_label'].value_counts()
                    cluster_names_map = {0: 'EXCELLENT', 1: 'AVERAGE', 2: 'PROBLEMATIC'}
                    cluster_counts.index = [cluster_names_map.get(i, f'Cluster {i}') for i in cluster_counts.index]

                    fig_cluster, ax = plt.subplots(figsize=(8, 5))
                    bars = ax.bar(cluster_counts.index, cluster_counts.values,
                                color=['#10b981', '#f59e0b', '#dc2626'])
                    ax.set_ylabel("Số lượng review", fontweight='bold')
                    ax.set_title("Phân bố số lượng review theo cụm", fontweight='bold', pad=20)
                    ax.bar_label(bars, fontweight='bold')
                    st.pyplot(fig_cluster)
                else:
                    st.info("⚠️ Dữ liệu chưa có nhãn cụm (`cluster_label`)")

            with col2:
                st.markdown("### 📋 Báo cáo mô hình")
                st.code("""📌 Model: KMeans Clustering
    Số cụm: 3
    Silhouette Score: 0.3247

    🏆 EXCELLENT:
    - Đánh giá tích cực cao
    - Từ khóa: "tuyệt vời", "hài lòng", "chế độ tốt"

    ⚖️ AVERAGE:
    - Cần cải thiện một số mặt
    - Từ khóa: "bình thường", "ổn", "trung lập"

    ⚠️ PROBLEMATIC:
    - Nhiều phàn nàn, áp lực
    - Từ khóa: "khó chịu", "toxic", "áp lực"
    """)

            st.markdown("### 🎯 Đặc điểm chi tiết từng cụm")
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("""
                <div class="metric-card" style="background: linear-gradient(135deg, #065f46 0%, #047857 100%); border-color: #10b981; color: #d1fae5;">
                    <h4 style="color: #6ee7b7;">🏆 EXCELLENT</h4>
                    <ul style="color: #d1fae5;">
                        <li>Rating trung bình: 4.2-5.0</li>
                        <li>Từ khóa tích cực cao</li>
                        <li>Chế độ tốt</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown("""
                <div class="metric-card" style="background: linear-gradient(135deg, #92400e 0%, #b45309 100%); border-color: #f59e0b; color: #fef3c7;">
                    <h4 style="color: #fcd34d;">⚖️ AVERAGE</h4>
                    <ul style="color: #fef3c7;">
                        <li>Rating trung bình: 3.0-4.1</li>
                        <li>Cần cải thiện một số mặt</li>
                        <li>Có cả tích cực & tiêu cực</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

            with col3:
                st.markdown("""
                <div class="metric-card" style="background: linear-gradient(135deg, #991b1b 0%, #b91c1c 100%); border-color: #dc2626; color: #fecaca;">
                    <h4 style="color: #fca5a5;">⚠️ PROBLEMATIC</h4>
                    <ul style="color: #fecaca;">
                        <li>Rating trung bình: 1.0-2.9</li>
                        <li>Nhiều vấn đề về quản lý</li>
                        <li>Cần cải thiện cấp thiết</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

            if os.path.exists('clustering/cluster_visualization.png'):
                st.markdown("### 📊 Trực quan hóa cụm")
                st.image('clustering/cluster_visualization.png', use_container_width=True)
            else:
                st.info("💡 Biểu đồ trực quan hóa cụm chưa có sẵn")
