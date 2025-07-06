import pandas as pd
import streamlit as st
import os
import difflib
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack, csr_matrix
import joblib

# Import sentiment analysis functions với error handling
try:
    from project_final import *
except ImportError as e:
    st.error(f"❌ Không thể import sentiment analysis module: {e}")
except Exception as e:
    st.error(f"❌ Lỗi khi import: {e}")

def sentiment_analysis_app(choice_lv2_clean, df_reviews):
    """Main function cho Sentiment Analysis app"""
    
    # Load models theo pattern từ file gốc
    scaler = StandardScaler(with_mean=False)
    
    try:
        vectorizer = joblib.load("sentiment/tfidf_vectorizer.pkl")
        model_final = joblib.load("sentiment/stacking.pkl")
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
            display_cols = ['Company Name', 'What I liked', 'Suggestions for improvement']
            with col1:
                st.markdown("**🔝 Top 3 đánh giá đầu tiên:**")
                st.dataframe(df_reviews[display_cols].head(3), use_container_width=True)
            with col2:
                st.markdown("**🔚 3 đánh giá cuối cùng:**")
                st.dataframe(df_reviews[display_cols].tail(3), use_container_width=True)

        with tab2:
            st.markdown("### ☁️ Trực quan hóa WordCloud toàn bộ review")
            
            if 'clean_advance_text' in df_reviews.columns:
                with st.spinner('Đang tạo WordCloud...'):
                    try:
                        keywords = get_key_words(df_reviews['clean_advance_text'].dropna())
                        fig_wc = check_wordcloud(keywords, 'Reviews')
                        st.pyplot(fig_wc, use_container_width=True)
                    except Exception as e:
                        st.error(f"❌ Không thể tạo WordCloud: {e}")
                        st.info("💡 Vui lòng kiểm tra dữ liệu text")
            else:
                st.warning("⚠️ Không tìm thấy cột 'clean_advance_text' trong dữ liệu")

        with tab3:
            st.markdown("### 🤖 Các mô hình đã huấn luyện và so sánh")
            
            # Bảng so sánh theo pattern từ file gốc
            st.markdown("""
                | Mô hình             | Accuracy | Ưu điểm                           | Nhược điểm                     |
                |---------------------|----------|-----------------------------------|--------------------------------|
                | Naive Bayes         | 0.8346   | Nhanh, đơn giản                   | Độ chính xác thấp              |
                | Logistic Regression | 0.9342   | Dễ triển khai, giải thích được    | Không xử lý phi tuyến tốt      |
                | SVM                 | 0.9469   | Phân biệt tốt                     | Tốn tài nguyên, chậm           |
                | Random Forest       | 0.9563   | Chính xác cao, chống overfit tốt  | Có thể hơi chậm khi scale lớn  |
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
                accuracies = [0.8346, 0.9342, 0.9469, 0.9563, 0.9695]
                
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
                st.code('''📌📌 Cross-Validation Accuracy: 0.9695 (+/- 0.0034)
                            📊 Classification Report for Stacking Model:
                                          precision    recall  f1-score   support
                            
                                negative       0.97      0.98      0.98       603
                                 neutral       0.98      1.00      0.99       617
                                positive       0.98      0.95      0.97       616
                            
                                accuracy                           0.98      1836
                               macro avg       0.98      0.98      0.98      1836
                            weighted avg       0.98      0.98      0.98      1836''')
                
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
            st.info("- tfidf_vectorizer.pkl")
            st.info("- stacking.pkl")
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

                    keywords = get_key_words([process_advance_text])
                    fig = check_wordcloud(keywords, 'Content')
                    if fig:
                        st.pyplot(fig, use_container_width=True)
                    else:
                        st.info("Không có từ khóa để hiển thị.")
                        
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
            
            lower_company_list = [c.lower() for c in company_list]
            search_text_lower = search_text.lower()

            matched_lowers = difflib.get_close_matches(search_text_lower, lower_company_list, n=10, cutoff=0.3)

            matched_companies = [company_list[lower_company_list.index(m)] for m in matched_lowers]

            if matched_companies:
                selected_company = st.selectbox("Chọn công ty phù hợp:", matched_companies)
            else:
                selected_company = None
                st.warning("❌ Không tìm thấy công ty phù hợp.")


        if selected_company:
            st.success(f"✅ Đang hiển thị thông tin cho: {selected_company}")

            df_company = df_reviews[df_reviews['Company Name'] == selected_company].copy()

            # 1. Tổng quan
            st.markdown(f"**Số lượng đánh giá:** {len(df_company)}")

            # 2. Tỷ lệ cảm xúc
            sentiment_counts = df_company['Setiment_FN'].value_counts(normalize=True).mul(100).round(2)
            st.write("### 📊 Tỷ lệ cảm xúc:")
            st.bar_chart(sentiment_counts)

            # 3. WordCloud
            col1, col2 = st.columns(2)
            with col1:
                liked_key_words = get_key_words(df_company['What I liked_procced'])
                fig_liked = check_wordcloud(liked_key_words, 'What I liked')
                if fig_liked:
                    st.pyplot(fig_liked, use_container_width=True)
                else:
                    st.info("Không có từ khóa để hiển thị.")

            with col2:
                suggest_key_words = get_key_words(df_company['Suggestions for improvement_procced'])
                fig_suggestions = check_wordcloud(suggest_key_words, 'Suggestions for improvement')
                if fig_suggestions:
                    st.pyplot(fig_suggestions, use_container_width=True)
                else:
                    st.info("Không có từ khóa để hiển thị.")

            # 4. Top từ khóa theo cảm xúc
            col1, col2 = st.columns(2)
            with col1:
                st.write("#### 🔴 Từ tiêu cực:")
                neg_df = df_company[df_company['Setiment_FN'] == 'negative']
                suggest_key_words = get_key_words(neg_df['Suggestions for improvement_procced'])
                fig_negative = check_wordcloud(suggest_key_words, 'negative')
                if fig_negative:
                    st.pyplot(fig_negative, use_container_width=True)
                else:
                    st.info("Không có từ khóa để hiển thị.")

            with col2:
                st.write("#### 🟢 Từ tích cực:")
                pos_df = df_company[df_company['Setiment_FN'] == 'positive']
                liked_key_words = get_key_words(pos_df['What I liked_procced'])
                fig_positive = check_wordcloud(liked_key_words, 'positive')
                if fig_positive:
                    st.pyplot(fig_positive, use_container_width=True)
                else:
                    st.info("Không có từ khóa để hiển thị.")

            # 5. Danh sách review (có thể ẩn/hiện)
            with st.expander("📄 Danh sách đánh giá (ẩn/hiện)"):
                st.dataframe(df_company[['What I liked', 'Suggestions for improvement', 'Setiment_FN']])