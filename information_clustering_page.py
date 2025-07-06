
import streamlit as st
import os
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack
from sklearn.decomposition import TruncatedSVD
import joblib
import matplotlib.pyplot as plt
import difflib

# Import sentiment analysis functions với error handling
try:
    from project_final import *
except ImportError as e:
    st.error(f"❌ Không thể import module: {e}")
except Exception as e:
    st.error(f"❌ Lỗi khi import: {e}")

def information_clustering_app(choice_lv2_clean, df_reviews):
    """Main function cho Information Clustering app"""
    
    # Load clustering models theo pattern từ file gốc
    scaler = joblib.load('clustering/cluster_standardScaler.pkl')
    cluster_names = ['Ít hài lòng', 'Hài lòng']
    
    try:
        liked_model = joblib.load("clustering/liked_model.pkl")
        suggested_model = joblib.load("clustering/suggested_model.pkl")
        svd_liked = joblib.load("clustering/svd_liked.pkl")
        svd_suggested = joblib.load("clustering/svd_suggested.pkl")
        models_loaded = True
    except FileNotFoundError as e:
        st.error(f"❌ Không tìm thấy file model: {e}")
        st.info("💡 Vui lòng kiểm tra các file sau có tồn tại:")
        st.info("- clustering/sentence_bert.pkl")
        st.info("- clustering/best_liked.pkl")
        st.info("- clustering/best_suggested.pkl")
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
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="metric-card" style="background: linear-gradient(135deg, #065f46 0%, #047857 100%); border-color: #10b981; color: #d1fae5;">
                <h3 style="color: #6ee7b7;">🏆 Hài lòng</h3>
                <p>Công ty xuất sắc với đánh giá rất tích cực</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card" style="background: linear-gradient(135deg, #92400e 0%, #b45309 100%); border-color: #f59e0b; color: #fef3c7;">
                <h3 style="color: #fcd34d;">⚖️ Ít hài lòng</h3>
                <p>Công ty có nhiều vấn cần cải thiện</p>
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
                columns_to_display = ['Company Name', 'What I liked', 'Suggestions for improvement'] + num_cols
                st.dataframe(df_reviews[columns_to_display].head(3), use_container_width=True)

            with col2:
                st.markdown("**🔚 3 đánh giá cuối cùng:**")
                columns_to_display = ['Company Name', 'What I liked', 'Suggestions for improvement'] + num_cols
                st.dataframe(df_reviews[columns_to_display].tail(3), use_container_width=True)

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

            if 'clean_advance_text' in df_reviews.columns:
                with st.spinner('Đang tạo WordCloud...'):
                    try:
                        keywords = get_key_words(df_reviews['clean_advance_text'].dropna())
                        fig_wc = check_wordcloud(keywords, 'Reviews')
                        st.pyplot(fig_wc, use_container_width=True)
                    except Exception as e:
                        st.error(f"❌ Không thể tạo WordCloud: {e}")
            else:
                st.warning("⚠️ Không tìm thấy cột 'clean_advance_text' trong dữ liệu")

            st.markdown("### 🔧 Quá trình tiền xử lý văn bản")
            st.markdown("""
            <div class="info-box">
                <h4>📝 Các bước tiền xử lý:</h4>
                <ul>
                    <li>🧹 Làm sạch văn bản (loại bỏ ký tự đặc biệt, số)</li>
                    <li>✂️ Phân đoạn câu theo ngữ nghĩa</li>
                    <li>🔤 Chuẩn hóa chữ hoa/thường</li>
                    <li>🚫 Loại bỏ stopwords</li>
                    <li>📊 Vector hóa bằng SBERT</li>
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

            st.markdown("""
                ##### 📌 1. Xử lý dữ liệu văn bản & số
                    - **Làm sạch và điền giá trị thiếu** cho 2 cột:
                      - `'What I liked_procced'`
                      - `'Suggestions for improvement_procced'`
                    - **Xử lý dữ liệu số** (Salary, Training, Culture,...) bằng StandardScaler
                
                ##### 📌 2. Sinh embedding Sentence-BERT
                    - Dùng mô hình `paraphrase-multilingual-mpnet-base-v2`
                    - Hỗ trợ tiếng Việt tốt
                    - Tối ưu bằng batch và GPU
                
                ##### 📌 3. Giảm chiều bằng TruncatedSVD
                    - Kết hợp embedding + dữ liệu số → giảm chiều
                    - Giúp tăng tốc và tránh curse of dimensionality
                    - Lưu lại tỷ lệ variance explained
                
                ##### 📌 4. So sánh mô hình clustering
                    - **Chạy 3 thuật toán:** KMeans, Agglomerative, DBSCAN
                    - **Đánh giá qua 3 metric:**
                      - Silhouette Score
                      - Davies-Bouldin Score
                      - Calinski-Harabasz Score
                    - **Chọn mô hình tốt nhất** cho từng phần (liked / suggested)
                
                ##### 📌 5. Lưu kết quả phân cụm
                    - Gán `liked_cluster` và `suggested_cluster` vào DataFrame
                    - Xuất ra file Excel
            """)

        with tab4:
            st.markdown("### 📈 Kết quả phân cụm")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("### 📊 What I liked")

                if 'liked_cluster' in df_reviews.columns:
                    cluster_counts = df_reviews['liked_cluster'].value_counts()
                    cluster_names_map = {0: 'Ít hài lòng', 1: 'Hài lòng'}
                    cluster_counts.index = [cluster_names_map.get(i, f'Cluster {i}') for i in cluster_counts.index]

                    fig_cluster, ax = plt.subplots(figsize=(8, 5))
                    bars = ax.bar(cluster_counts.index, cluster_counts.values,
                                color=['#10b981', '#f59e0b', '#dc2626'])
                    ax.set_ylabel("Số lượng review", fontweight='bold')
                    ax.set_title("Phân bố số lượng review theo cụm", fontweight='bold', pad=20)
                    ax.bar_label(bars, fontweight='bold')
                    st.pyplot(fig_cluster)
                else:
                    st.info("⚠️ Dữ liệu chưa có nhãn cụm (`liked_cluster`)")

            with col2:
                st.markdown("### 📊 Suggestions for improvement")

                if 'suggested_cluster' in df_reviews.columns:
                    cluster_counts = df_reviews['suggested_cluster'].value_counts()
                    cluster_names_map = {0: 'Ít hài lòng', 1: 'Hài lòng'}
                    cluster_counts.index = [cluster_names_map.get(i, f'Cluster {i}') for i in cluster_counts.index]

                    fig_cluster, ax = plt.subplots(figsize=(8, 5))
                    bars = ax.bar(cluster_counts.index, cluster_counts.values,
                                  color=['#10b981', '#f59e0b', '#dc2626'])
                    ax.set_ylabel("Số lượng review", fontweight='bold')
                    ax.set_title("Phân bố số lượng review theo cụm", fontweight='bold', pad=20)
                    ax.bar_label(bars, fontweight='bold')
                    st.pyplot(fig_cluster)
                else:
                    st.info("⚠️ Dữ liệu chưa có nhãn cụm (`suggested_cluster`)")

            with col3:
                st.markdown("### 📋 Báo cáo mô hình")

                st.markdown("""
                ##### 📌 Model: KMeans Clustering
                    - **Số cụm:** 3
                    - **Silhouette Score:** 0.3247
                
                ##### 🏆 Hài lòng
                    - **Đánh giá tích cực cao**
                    - **Từ khóa:** "tuyệt vời", "hài lòng", "chế độ tốt"
                
                ##### ⚖️ Ít hài lòng
                    - **Cần cải thiện một số mặt**
                    - **Từ khóa:** "bình thường", "ổn", "trung lập"
                """)

            st.markdown("### 🎯 Đặc điểm chi tiết từng cụm")
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("""
                <div class="metric-card" style="background: linear-gradient(135deg, #065f46 0%, #047857 100%); border-color: #10b981; color: #d1fae5;">
                    <h4 style="color: #6ee7b7;">🏆 Hài lòng</h4>
                    <ul style="color: #d1fae5;">
                        <li>Rating trung bình: > 4.0 </li>
                        <li>Từ khóa tích cực cao</li>
                        <li>Chế độ tốt</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown("""
                    <div class="metric-card" style="background: linear-gradient(135deg, #92400e 0%, #b45309 100%); border-color: #f59e0b; color: #fef3c7;">
                        <h4 style="color: #fcd34d;">⚖️ Ít hài lòng</h4>
                        <ul style="color: #fef3c7; list-style-type: none; padding-left: 0;">
                            <li>Rating trung bình: < 4.0</li>
                            <li>Cần cải thiện một số mặt</li>
                            <li>Có cả tích cực & tiêu cực</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)

            if os.path.exists('clustering/cluster_visualization.png'):
                st.markdown("### 📊 Trực quan hóa cụm")
                st.image('clustering/cluster_visualization.png', use_container_width=True)
            else:
                st.info("💡 Biểu đồ trực quan hóa cụm chưa có sẵn")
                
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

            liked_text = st.text_area(label="What I liked")
            suggested_text = st.text_area(label="Suggestions for improvement")

            salary = st.slider("Salary & benefits", 1, 5, 5)
            training = st.slider("Training & learning", 1, 5, 5)
            cares = st.slider("Management cares about me", 1, 5, 5)
            fun = st.slider("Culture & fun", 1, 5, 5)
            workspace = st.slider("Office & workspace", 1, 5, 5)

            if liked_text.strip() != '':
                try:
                    print(liked_text)
                    process_text = process_basic_text(liked_text)
                    print("process_text ok.")
                    liked_embedding = embedding_model.encode([process_text],batch_size=32, show_progress_bar=True,convert_to_numpy=True)
                    print('liked_embedding ok.')
                    X_num = scaler.transform([[salary, training, cares, fun, workspace]])
                    print('scaler number ok.')

                    # Ghép embedding với dữ liệu số
                    liked_all = np.hstack([liked_embedding, X_num])
                    print('hstack ok.')
                    liked_reduced = svd_liked.transform(liked_all)
                    print('SVD ok')
                    liked_cluster = liked_model.predict(liked_reduced)[0]
                    print('cluster ok.')
                    st.success(f"🎯 What I liked: Đánh giá này thuộc nhóm **{cluster_names[liked_cluster]}**")

                    keywords = get_key_words([process_text])
                    fig = check_wordcloud(keywords, 'What I liked')
                    if fig:
                        st.pyplot(fig, use_container_width=True)
                    else:
                        st.info("Không có từ khóa để hiển thị.")

                except Exception as e:
                    st.error(f"❌ Có lỗi xảy ra trong quá trình phân tích: {str(e)}")
                    st.info("💡 Vui lòng thử lại hoặc kiểm tra lại nội dung đầu vào.")

            if suggested_text.strip() != '':
                try:
                    print(suggested_text)
                    process_text = process_basic_text(suggested_text)
                    print("process_text ok.")
                    suggested_embedding = embedding_model.encode([process_text],batch_size=32, show_progress_bar=True,convert_to_numpy=True)
                    print('liked_embedding ok.')
                    X_num = scaler.transform([[salary, training, cares, fun, workspace]])
                    print(X_num)
                    print('scaler number ok.')

                    # Ghép embedding với dữ liệu số
                    suggested_all = np.hstack([suggested_embedding, X_num])
                    print('hstack ok.')
                    suggested_reduced = svd_suggested.transform(suggested_all)
                    print('SVD ok')
                    suggested_cluster = suggested_model.predict(suggested_reduced)[0]
                    print('cluster ok.')
                    st.success(f"🎯 Suggestions for improvement: Đánh giá này thuộc nhóm **{cluster_names[suggested_cluster]}**")

                    keywords = get_key_words([process_text])
                    fig = check_wordcloud(keywords, 'Suggestions for improvement')
                    if fig:
                        st.pyplot(fig, use_container_width=True)
                    else:
                        st.info("Không có từ khóa để hiển thị.")

                except Exception as e:
                    st.error(f"❌ Có lỗi xảy ra trong quá trình phân tích: {str(e)}")
                    st.info("💡 Vui lòng thử lại hoặc kiểm tra lại nội dung đầu vào.")


            st.markdown("---")

            # Phần phân tích theo phân cụm - giữ nguyên pattern từ file gốc
            st.subheader("🎯 Phân tích theo phân cụm")

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
                    if search_text != '':
                        st.warning("❌ Không tìm thấy công ty phù hợp.")

            if selected_company:
                st.success(f"✅ Đang hiển thị thông tin cho: {selected_company}")

                df_company = df_reviews[df_reviews['Company Name'] == selected_company].copy()

                # Chọn loại phân cụm
                cluster_type = st.radio("Chọn loại phân cụm:", ['Liked Cluster', 'Suggested Cluster'])

                if cluster_type == 'Liked Cluster':
                    cluster_data = 'What I liked'
                    cluster_column = 'liked_cluster'
                    cluster_title = "cụm Liked"
                else:
                    cluster_data = 'Suggestions for improvement'
                    cluster_column = 'suggested_cluster'
                    cluster_title = "cụm Suggested"

                # Kiểm tra xem có cột cluster không
                if cluster_column in df_company.columns:
                    cluster_list = df_company[cluster_column].dropna().unique().tolist()
                    cluster_list.sort()
                    cluster_mapping = {cluster_names[i] : cluster_list[i] for i in range(len(cluster_list))}

                    selected_cluster = st.selectbox(f"Chọn {cluster_title} để phân tích:", list(cluster_mapping.keys()))

                    if selected_cluster is not None:
                        st.success(f"✅ Đang hiển thị thông tin cho {cluster_title}: {selected_cluster}")

                        df_cluster = df_company[df_company[cluster_column] == cluster_mapping.get(selected_cluster)].copy()

                        # 1. Tổng quan
                        st.markdown(f"**Số lượng đánh giá:** {len(df_cluster)}")

                        # 2. Tỷ lệ cảm xúc
                        sentiment_counts = df_cluster['Setiment_FN'].value_counts(normalize=True).mul(100).round(2)
                        st.write("### 📊 Tỷ lệ cảm xúc:")
                        st.bar_chart(sentiment_counts)

                        # 3. WordCloud
                        liked_key_words = get_key_words(df_cluster[cluster_data + '_procced'])
                        fig_liked = check_wordcloud(liked_key_words, cluster_data)
                        if fig_liked:
                            st.pyplot(fig_liked, use_container_width=True)
                        else:
                            st.info("Không có từ khóa để hiển thị.")

                        # 4. Top từ khóa theo cảm xúc
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("#### 🔴 Từ tiêu cực:")
                            neg_df = df_cluster[df_cluster['Setiment_FN'] == 'negative']
                            suggest_key_words = get_key_words(neg_df['Suggestions for improvement_procced'])
                            fig_negative = check_wordcloud(suggest_key_words, 'negative')
                            if fig_negative:
                                st.pyplot(fig_negative, use_container_width=True)
                            else:
                                st.info("Không có từ khóa để hiển thị.")

                        with col2:
                            st.write("#### 🟢 Từ tích cực:")
                            pos_df = df_cluster[df_cluster['Setiment_FN'] == 'positive']
                            liked_key_words = get_key_words(pos_df['What I liked_procced'])
                            fig_positive = check_wordcloud(liked_key_words, 'positive')
                            if fig_positive:
                                st.pyplot(fig_positive, use_container_width=True)
                            else:
                                st.info("Không có từ khóa để hiển thị.")

                        # 5. Danh sách review (có thể ẩn/hiện)
                        with st.expander("📄 Danh sách đánh giá (ẩn/hiện)"):
                            cluster_mapping = {cluster_list[i] : cluster_names[i] for i in range(len(cluster_list))}
                            df_display = df_cluster.copy()
                            df_display[cluster_column] = df_display[cluster_column].map(cluster_mapping)
                            columns_to_display = [cluster_data, 'Setiment_FN'] + num_cols + [cluster_column]
                            st.dataframe(df_display[columns_to_display])

                else:
                    st.warning(f"❌ Không tìm thấy cột '{cluster_column}' trong dữ liệu. Vui lòng kiểm tra lại dữ liệu đầu vào.")