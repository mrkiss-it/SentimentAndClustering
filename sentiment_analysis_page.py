import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import os
import difflib
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

def sentiment_analysis_app(choice_lv2_clean, df_reviews):
    """Main function cho Sentiment Analysis app"""
    
    # Load models theo pattern từ file gốc
    scaler = MinMaxScaler()
    
    try:
        vectorizer = joblib.load("sentiment/tfidf_vectorizer2.pkl")
        model_final = joblib.load("sentiment/stacking_model.pkl")
        models_loaded = True
    except FileNotFoundError as e:
        st.error(f"❌ Không tìm thấy file model: {e}")
        st.info("💡 Vui lòng kiểm tra các file sau có tồn tại:")
        st.info("- sentiment/tfidf_vectorizer2.pkl")
        st.info("- sentiment/stacking_model.pkl")
        vectorizer = None
        model_final = None
        models_loaded = False
    except Exception as e:
        st.error(f"❌ Lỗi khi load models: {e}")
        vectorizer = None
        model_final = None
        models_loaded = False

    if choice_lv2_clean == "Business Objective":
        st.markdown('<h1 class="section-header">🎯 Mục tiêu phân tích cảm xúc</h1>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="info-card">
                <h3>📝 Yêu cầu</h3>
                <p>Các công ty đang nhận nhiều đánh giá từ ITviec</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="info-card">
                <h3>🎯 Mục tiêu</h3>
                <p>Phân tích cảm xúc: tích cực, tiêu cực, trung lập</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="info-card">
                <h3>💼 Ứng dụng</h3>
                <p>Đánh giá độ hài lòng nhân viên, cải thiện hình ảnh</p>
            </div>
            """, unsafe_allow_html=True)

    elif choice_lv2_clean == "Build Project":
        st.markdown('<h1 class="section-header">🔧 Xây dựng mô hình phân tích cảm xúc</h1>', unsafe_allow_html=True)

        # Tab layout
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Dữ liệu", "☁️ WordCloud", "🤖 Mô hình", "📈 Kết quả"])
        
        with tab1:
            st.markdown("### 📋 Dữ liệu mẫu từ review")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**🔝 Top 3 đánh giá đầu tiên:**")
                display_cols = ['Company Name', 'reviews_text'] if 'reviews_text' in df_reviews.columns else df_reviews.columns[:2]
                st.dataframe(df_reviews[display_cols].head(3), use_container_width=True)
            
            with col2:
                st.markdown("**🔚 3 đánh giá cuối cùng:**")
                st.dataframe(df_reviews[display_cols].tail(3), use_container_width=True)

        with tab2:
            st.markdown("### ☁️ Trực quan hóa WordCloud toàn bộ review")
            
            if 'clean_advance_text2' in df_reviews.columns:
                with st.spinner('Đang tạo WordCloud...'):
                    try:
                        fig_wc = check_wordcloud(df_reviews['clean_advance_text2'].dropna(), 'Reviews')
                        st.pyplot(fig_wc, use_container_width=True)
                    except Exception as e:
                        st.error(f"❌ Không thể tạo WordCloud: {e}")
                        st.info("💡 Vui lòng kiểm tra dữ liệu text")
            else:
                st.warning("⚠️ Không tìm thấy cột 'clean_advance_text2' trong dữ liệu")

        with tab3:
            st.markdown("### 🤖 Các mô hình đã huấn luyện và so sánh")
            
            # Bảng so sánh theo pattern từ file gốc
            st.markdown("""
        | Mô hình             | Accuracy | Ưu điểm                           | Nhược điểm                     |
        |---------------------|----------|-----------------------------------|--------------------------------|
        | Naive Bayes         | 0.8237   | Nhanh, đơn giản                   | Độ chính xác thấp              |
        | Logistic Regression | 0.9448   | Dễ triển khai, giải thích được    | Không xử lý phi tuyến tốt      |
        | SVM                 | 0.9529   | Phân biệt tốt                     | Tốn tài nguyên, chậm           |
        | Random Forest       | 0.9643   | Chính xác cao, chống overfit tốt  | Có thể hơi chậm khi scale lớn  |
        """)
            
            st.markdown("""
            <div class="info-box">
                <h4>🏆 Mô hình Stacking</h4>
                <p>Mô hình <strong>StackingClassifier</strong> được xây dựng bằng cách kết hợp 3 mô hình mạnh nhất:</p>
                <ul>
                    <li>🎯 Logistic Regression</li>
                    <li>🎯 SVM</li>
                    <li>🎯 Random Forest</li>
                </ul>
                <p>Sau đó, một <strong>Logistic Regression</strong> được dùng làm <strong>meta-model</strong> để tổng hợp kết quả.</p>
                <p>✅ Sử dụng <code>passthrough=True</code> giúp meta-model thấy cả đặc trưng gốc lẫn kết quả trung gian.</p>
            </div>
            """, unsafe_allow_html=True)

        with tab4:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("### 📊 So sánh Accuracy")
                model_names = ["Naive Bayes", "Logistic Regression", "SVM", "Random Forest", "Stacking"]
                accuracies = [0.8237, 0.9448, 0.9529, 0.9643, 0.9804]
                
                fig_acc, ax = plt.subplots(figsize=(10, 6))
                bars = ax.bar(model_names, accuracies, 
                             color=['#6b7280', '#f59e0b', '#3b82f6', '#10b981', '#8b5cf6'])
                ax.set_ylim(0.8, 0.985)
                ax.set_ylabel("Accuracy", fontweight='bold')
                ax.set_title("So sánh độ chính xác giữa các mô hình", fontweight='bold', pad=20)
                ax.bar_label(bars, fmt="%.4f", padding=3, fontweight='bold')
                plt.xticks(rotation=15)
                plt.tight_layout()
                st.pyplot(fig_acc)
            
            with col2:
                st.markdown("### 📋 Báo cáo mô hình cuối cùng")
                st.code('''📌 Model: StackingClassifier
Cross-Validation Accuracy: 0.9804 (+/- 0.0029)
Classification Report:
              precision    recall  f1-score   support
    negative       0.98      0.98      0.98       742
     neutral       0.97      0.99      0.98       744
    positive       0.98      0.95      0.97       745

    accuracy                           0.98      2231
   macro avg       0.98      0.98      0.98      2231
weighted avg       0.98      0.98      0.98      2231''')
                
                # Hiển thị confusion matrix nếu có
                if os.path.exists('sentiment/Confusion Matrix -  Stacking.png'):
                    st.image('sentiment/Confusion Matrix -  Stacking.png', use_container_width=True)
                else:
                    st.info("💡 Confusion matrix image không tìm thấy")

    elif choice_lv2_clean == "New Prediction":
        st.markdown('<h1 class="section-header">🚀 Dự đoán cảm xúc mới</h1>', unsafe_allow_html=True)
        
        if not models_loaded:
            st.error("❌ Không thể load models. Vui lòng kiểm tra lại file models.")
            st.info("💡 Cần các file sau trong thư mục sentiment/:")
            st.info("- tfidf_vectorizer2.pkl")
            st.info("- stacking_model.pkl")
        else:
            # Phần nhập text theo pattern từ file gốc
            text = st.text_area(label="Nhập nội dung của bạn:")
            
            if text != '':
                try:
                    process_text = process_basic_text(text)
                    lang = detect_lang_safe(process_text)

                    pos_w, neg_w, pos_e, neg_e, total_we, ratio_all = calc_sentiment_features(process_text)
                    print(pos_w, neg_w, pos_e, neg_e, total_we, ratio_all)

                    split_txt = split_sentences_by_meaning(process_text, lang)
                    process_advance_text = process_split_text(split_txt, lang)
                    print(process_advance_text)

                    X_tfidf = vectorizer.transform([process_advance_text])
                    X_num = scaler.fit_transform([[pos_w, neg_w, pos_e, neg_e, total_we, ratio_all]])
                    X = hstack([X_tfidf, csr_matrix(X_num)])

                    y_pred = model_final.predict(X)[0]
                    st.write(f"Kết quả dự đoán là: {y_pred}")
                    
                    if y_pred == 'positive':
                        st.write(", ".join([x.strip() for x in pos_words if x.strip() != "" and x.lower() in process_text.lower()]))
                    elif y_pred == 'negative':
                        st.write(", ".join([x.strip() for x in neg_words if x.strip() != "" and x.lower() in process_text.lower()]))

                    fig = check_wordcloud([process_advance_text], 'Content')
                    st.pyplot(fig, use_container_width=True)
                        
                except Exception as e:
                    st.error(f"❌ Lỗi khi phân tích: {e}")
                    st.info("💡 Vui lòng kiểm tra lại nội dung đầu vào hoặc liên hệ admin.")

        st.markdown("---")
        
        # Phần phân tích theo công ty - giữ nguyên pattern từ file gốc
        st.subheader("🏢 Phân tích theo tên công ty")

        company_list = df_reviews['Company Name'].dropna().unique().tolist()
        company_list.sort()

        search_type = st.radio("Chọn cách tìm công ty:", ['Chọn từ danh sách', 'Nhập tên gần đúng'])

        if search_type == 'Chọn từ danh sách':
            selected_company = st.selectbox("Chọn công ty", company_list)
        else:
            search_text = st.text_input("Nhập tên công ty (gần đúng):")
            matched_companies = difflib.get_close_matches(search_text, company_list, n=5, cutoff=0.3)
            if matched_companies:
                selected_company = st.selectbox("Chọn công ty phù hợp:", matched_companies)
            else:
                selected_company = None
                st.warning("❌ Không tìm thấy công ty phù hợp.")

        if selected_company:
            st.success(f"✅ Đang hiển thị thông tin cho: {selected_company}")

            df_company = df_reviews[df_reviews['Company Name'] == selected_company]

            # 1. Tổng quan
            st.markdown(f"**Số lượng đánh giá:** {len(df_company)}")

            # 2. Tỷ lệ cảm xúc
            sentiment_counts = df_company['Pred_FN'].value_counts(normalize=True).mul(100).round(2)
            st.write("### 📊 Tỷ lệ cảm xúc:")
            st.bar_chart(sentiment_counts)

            # 3. WordCloud
            fig_wc = check_wordcloud(df_company['clean_advance_text2'], 'Reviews')
            st.pyplot(fig_wc, use_container_width=True)

            # 4. Top từ khóa theo cảm xúc
            col1, col2 = st.columns(2)
            with col1:
                st.write("#### 🔴 Từ tiêu cực:")
                neg_df = df_company[df_company['Pred_FN'] == 'negative']
                neg_texts = " ".join(neg_df['clean_advance_text2'].dropna().astype(str))

                if neg_texts.strip():
                    try:
                        fig_neg, ax = plt.subplots(figsize=(8, 4))
                        wc_neg = WordCloud(width=800, height=400, background_color='white').generate(neg_texts)
                        ax.imshow(wc_neg, interpolation='bilinear')
                        ax.axis("off")
                        st.pyplot(fig_neg)
                    except ValueError as e:
                        st.warning("❌ Không đủ từ để tạo WordCloud tiêu cực.")
                else:
                    st.info("💬 Không có review tiêu cực nào.")

            with col2:
                st.write("#### 🟢 Từ tích cực:")
                pos_df = df_company[df_company['Pred_FN'] == 'positive']
                pos_texts = " ".join(pos_df['clean_advance_text2'].dropna().astype(str))

                if pos_texts.strip():
                    try:
                        fig_pos, ax = plt.subplots(figsize=(8, 4))
                        wc_pos = WordCloud(width=800, height=400, background_color='white').generate(pos_texts)
                        ax.imshow(wc_pos, interpolation='bilinear')
                        ax.axis("off")
                        st.pyplot(fig_pos)
                    except ValueError as e:
                        st.warning("❌ Không đủ từ để tạo WordCloud tích cực.")
                else:
                    st.info("💬 Không có review tích cực nào.")

            # 5. Danh sách review (có thể ẩn/hiện)
            with st.expander("📄 Danh sách đánh giá (ẩn/hiện)"):
                st.dataframe(df_company[['clean_basic_text', 'Pred_FN']])