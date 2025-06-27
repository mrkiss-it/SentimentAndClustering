import difflib
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse import hstack, csr_matrix
import joblib
from sentiment.sentiment_analysis import *

def check_wordcloud(data, col_name):
    text = " ".join(data)  # Gộp danh sách thành chuỗi
    wc = WordCloud(width=800, height=400, background_color='white').generate(text)

    # Tạo figure và vẽ WordCloud
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis("off")
    ax.set_title("WordCloud của " + col_name)

    return fig

# Đọc dữ liệu sản phẩm
if 'reviews' not in st.session_state:
    st.session_state.reviews = pd.read_excel('processed_reviews.xlsx')

df_reviews = st.session_state.reviews

###### GUI ######
st.image('channels4_banner.jpg', use_container_width=True) # phiên bản mới hơn

# Menu cấp 1
choice_lv1 = st.sidebar.selectbox('Menu', ['Tổng quan', 'Sentiment Analysis', 'Information Clustering'])

# Menu cấp 2
menu_lv2 = {
    'Tổng quan': ['Giới thiệu'],
    'Sentiment Analysis': ["Business Objective", "Build Project", "New Prediction"],
    'Information Clustering': ["Business Objective", "Build Project", "New Prediction"]
}
choice_lv2 = st.sidebar.selectbox(choice_lv1, menu_lv2.get(choice_lv1, []))

st.text(f"📌 Bạn đang ở: {choice_lv1} > {choice_lv2}")

st.title(f'{choice_lv2}')
# === Nội dung hiển thị theo từng mục ===
if choice_lv1 == 'Tổng quan':
    if choice_lv2 == 'Giới thiệu':
        st.markdown("""
**Sentiment Analysis** là quá trình sử dụng xử lý ngôn ngữ tự nhiên và học máy để phân tích cảm xúc trong các đánh giá, phản hồi từ người dùng (tích cực, tiêu cực, trung lập).
  
**Information Clustering** giúp phân nhóm các đánh giá để doanh nghiệp hiểu rõ họ thuộc nhóm nào → từ đó cải thiện và phát triển tốt hơn.
        
Ứng dụng thực tế: Dữ liệu từ [ITviec.com](https://itviec.com/) với các review từ ứng viên và nhân viên.
        """)

elif choice_lv1 == 'Sentiment Analysis':
    # Load models
    scaler = MinMaxScaler()
    vectorizer = joblib.load("sentiment/tfidf_vectorizer2.pkl")
    model_final = joblib.load("sentiment/stacking_model.pkl")

    if choice_lv2 == "Business Objective":
        st.subheader("🎯 Mục tiêu phân tích cảm xúc")
        st.markdown("""
- **Yêu cầu**: Các công ty đang nhận nhiều đánh giá từ ITviec.  
- Mục tiêu là **phân tích cảm xúc** các review này: tích cực, tiêu cực hay trung lập.
- Áp dụng trong đánh giá độ hài lòng nhân viên, cải thiện hình ảnh công ty.
        """)

    elif choice_lv2 == "Build Project":
        st.subheader("🔧 Xây dựng mô hình phân tích cảm xúc")

        st.write("##### 1. Dữ liệu mẫu từ review")
        st.dataframe(df_reviews[['Company Name', 'reviews_text']].head(3))
        st.dataframe(df_reviews[['Company Name', 'reviews_text']].tail(3))

        st.write("##### 2. Trực quan hóa WordCloud toàn bộ review")
        fig_wc = check_wordcloud(df_reviews['clean_advance_text2'], 'Reviews')
        if fig_wc:
            st.pyplot(fig_wc.figure)

        st.write("##### 3. Các mô hình đã huấn luyện và so sánh")
        st.markdown("""
    | Mô hình             | Accuracy | Ưu điểm                           | Nhược điểm                     |
    |---------------------|----------|-----------------------------------|--------------------------------|
    | Naive Bayes         | 0.8237   | Nhanh, đơn giản                   | Độ chính xác thấp              |
    | Logistic Regression | 0.9448   | Dễ triển khai, giải thích được    | Không xử lý phi tuyến tốt      |
    | SVM                 | 0.9529   | Phân biệt tốt                     | Tốn tài nguyên, chậm           |
    | Random Forest       | 0.9643   | Chính xác cao, chống overfit tốt  | Có thể hơi chậm khi scale lớn  |
    """)

        st.write("##### 4. Kết hợp mô hình (Stacking)")
        st.markdown("""
    Mô hình **StackingClassifier** được xây dựng bằng cách kết hợp 3 mô hình mạnh nhất:
    
    - 🎯 Logistic Regression
    - 🎯 SVM
    - 🎯 Random Forest
    
    Sau đó, một **Logistic Regression** được dùng làm **meta-model** để tổng hợp kết quả từ các mô hình con.
    
    ✅ Sử dụng `passthrough=True` giúp meta-model thấy cả đặc trưng gốc lẫn kết quả trung gian.
    
    **Ưu điểm:** kết hợp điểm mạnh của nhiều mô hình → tăng độ chính xác và khả năng tổng quát hóa.
    """)

        st.write("##### 5. So sánh Accuracy giữa các mô hình")
        import matplotlib.pyplot as plt
        model_names = ["Naive Bayes", "Logistic Regression", "SVM", "Random Forest", "Stacking"]
        accuracies = [0.8237, 0.9448, 0.9529, 0.9643, 0.9804]
        fig_acc, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(model_names, accuracies, color=['gray', 'orange', 'blue', 'green', 'purple'])
        ax.set_ylim(0.8, 0.985)
        ax.set_ylabel("Accuracy")
        ax.set_title("So sánh độ chính xác giữa các mô hình")
        ax.bar_label(bars, fmt="%.4f", padding=3)
        plt.xticks(rotation=15)
        plt.tight_layout()
        st.pyplot(fig_acc)

        st.write("##### 6. Báo cáo mô hình cuối cùng (Stacking)")
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

        st.image('sentiment/Confusion Matrix -  Stacking.png')



    elif choice_lv2 == "New Prediction":
        st.subheader("🚀 Dự đoán cảm xúc mới")
        text = st.text_area(label="Nhập nội dung của bạn:")
        if text != '':
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
            st.pyplot(fig.figure)


        st.markdown("---")
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
            st.pyplot(fig_wc.figure)

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

elif choice_lv1 == 'Information Clustering':
    # Load model
    scaler = MinMaxScaler()
    clustering_vectorizer = joblib.load("clustering/tfidf_vectorizer.pkl")
    clustering_model = joblib.load("clustering/best_prediction_model.pkl")
    cluster_names = ['EXCELLENT', 'AVERAGE', 'PROBLEMATIC']

    if choice_lv2 == "Business Objective":
        st.subheader("📌 Mục tiêu phân cụm thông tin")
        st.markdown("""
- **Yêu cầu**: Phân nhóm review theo nội dung và cảm nhận.
- Mỗi công ty biết mình thuộc cụm nào → đưa ra chiến lược cải thiện phù hợp.
        """)

    elif choice_lv2 == "Build Project":
        st.subheader("🏗️ Xây dựng mô hình phân cụm")
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
        st.pyplot(fig.figure)

        st.write("##### 5. Tổng kết")
        st.success("Model phân cụm đã sẵn sàng để phân loại các đánh giá mới vào 3 nhóm chính!")

    elif choice_lv2 == "New Prediction":
        st.subheader("🆕 Gom nhóm đánh giá mới")
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
            st.pyplot(fig.figure)



