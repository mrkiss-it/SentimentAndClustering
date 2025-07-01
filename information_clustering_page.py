
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
        st.markdown('<h1 class="section-header">🏗️ Xây dựng mô hình phân cụm</h1>', unsafe_allow_html=True)
        
        # Theo pattern từ file gốc
        st.markdown("""
        - Thuật toán được sử dụng: **KMeans** (đã được chọn làm model tốt nhất)
        - Quá trình xử lý:
            1. Tiền xử lý văn bản bằng TF-IDF
            2. Chuẩn hóa các đặc trưng số (rating, salary...)
            3. Kết hợp đặc trưng số và văn bản
            4. Phân cụm bằng KMeans
        """)

        st.write("##### 1. Dữ liệu đầu vào")
        st.dataframe(df_reviews[['Company Name', 'reviews_text']].head(3))
        st.dataframe(df_reviews[['Company Name', 'reviews_text']].tail(3))

        st.write("##### 2. Tiền xử lý văn bản")
        st.code("""
        # Quá trình tiền xử lý bao gồm:
        - Làm sạch văn bản
        - Phân đoạn câu (split sentences)
        - Xử lý ngôn ngữ tự nhiên
        - Vector hóa bằng TF-IDF
        """)

        st.write("##### 3. Xây dựng mô hình")
        st.code("""
        # Các bước chính:
        1. Khởi tạo MinMaxScaler cho các đặc trưng số
        2. Load TF-IDF vectorizer đã được huấn luyên
        3. Load mô hình KMeans đã được huấn luyện
        4. Kết hợp đặc trưng số và văn bản
        5. Phân cụm bằng KMeans
        """)

        st.write("##### 4. Đánh giá cụm")
        st.markdown("""
        - Các cụm được đặt tên: **EXCELLENT**, **AVERAGE**, **PROBLEMATIC**
        - Đánh giá chất lượng cụm bằng phương pháp Silhouette Score
        - Trực quan hóa bằng word cloud cho từng cụm
        """)

        # Hiển thị word cloud mẫu
        fig = check_wordcloud(df_reviews['clean_advance_text2'], 'Reviews')
        st.pyplot(fig, use_container_width=True)

        st.write("##### 5. Tổng kết")
        st.success("Model phân cụm đã sẵn sàng để phân loại các đánh giá mới vào 3 nhóm chính!")

    elif choice_lv2_clean == "New Prediction":
        st.markdown('<h1 class="section-header">🆕 Gom nhóm đánh giá mới</h1>', unsafe_allow_html=True)
        
        if not models_loaded:
            st.error("❌ Không thể load clustering models. Vui lòng kiểm tra lại file models.")
            st.info("💡 Cần các file sau trong thư mục clustering/:")
            st.info("- tfidf_vectorizer.pkl")
            st.info("- best_prediction_model.pkl")
        else:
            st.markdown("""
            - Nhập dữ liệu review mới → đưa vào mô hình clustering.
            - Mỗi review/công ty được gán vào 1 cụm → giúp hiểu nội dung tổng quát.
            """)
            
            text = st.text_area(label="Nhập nội dung của bạn:")

            rating = st.slider("Rating", 1, 5, 1)
            salary = st.slider("Salary & benefits", 1, 5, 1)
            training = st.slider("Training & learning", 1, 5, 1)
            cares = st.slider("Management cares about me", 1, 5, 1)
            fun = st.slider("Culture & fun", 1, 5, 1)
            workspace = st.slider("Office & workspace", 1, 5, 1)
            print(rating, salary, training, cares, fun, workspace)

            if text != '':
                try:
                    process_text = process_basic_text(text)
                    lang = detect_lang_safe(process_text)

                    split_txt = split_sentences_by_meaning(process_text, lang)
                    process_advance_text = process_split_text(split_txt, lang)
                    print(process_advance_text)

                    X_tfidf = clustering_vectorizer.transform([process_text])
                    X_num = scaler.fit_transform([[rating, salary, training, cares, fun, workspace]])
                    X = hstack([X_num, X_tfidf])

                    y_pred = clustering_model.predict(X)[0]
                    st.write(f"Đây là công ty: {cluster_names[y_pred]}")

                    fig = check_wordcloud([process_text], 'clean_text')
                    st.pyplot(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"❌ Có lỗi xảy ra trong quá trình phân tích: {str(e)}")
                    st.info("💡 Vui lòng thử lại hoặc kiểm tra lại nội dung đầu vào.")