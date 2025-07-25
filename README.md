# 📉 Dự Án Phân Tích Đánh Giá Nhân Viên

## 🌟 Giới Thiệu

Dự án này nhằm phân tích dữ liệu đánh giá nhân viên do chính họs cung cấp. Mục tiêu là khai thác những điểm tích cực, góp ý cải thiện và các vấn đề nổi bật thông qua hai nhiệm vụ chính:

1. **Phân tích cảm xúc (Sentiment Analysis)**  
    > **Mục tiêu:** Phân tích cảm xúc từ review của ứng viên/nhân viên trên ITviec

2. **Phân cụm nội dung đánh giá (Clustering)**  
   > **Mục tiêu:** Nhóm hoá review theo nội dung tương đồng để doanh nghiệp biết mình thuộc nhóm nhận được đánh giá nào
   
---

## 🧠 Phân Tích Cảm Xúc (Sentiment Analysis)

### Mô tả:

Sử dụng mô hình ViSoBERT để gán nhãn cảm xúc cho hai trường dữ liệu chính: `What I liked` và `Suggestions for improvement`.

### Các bước:

* Tiền xử lý văn bản (loại stopword, tạo token)
* Dự đoán nhãn cảm xúc (positive, negative, neutral)
* Trích xuất đặc trưng bổ sung: emoji, từ khóa, tỷ lệ cảm xúc

### Kết Quả Nổi Bật:

* Hơn **50%** đánh giá mang cảm xúc **tích cực**
* Khoảng **30%** phợ biểu đồ **tiêu cực**
* Còn lại là trung tính

### Hàm:

* `visobert_sentiment()`
* `calc_sentiment_features()`

---
## 🔎 Phân Cụm Đánh Giá (Clustering)

### Mô tả:

Tự động nhận diện các chủ đề bằng cách nhóm các đoạn đánh giá có nội dung tương đồng

### Pipeline:

1. TF-IDF vectorization
2. Sử dụng thuật toán: KMeans (với k=2 theo Silhouette Score), DBSCAN, Agglomerative
3. Trực quan hóa cụm bằng PCA và t-SNE, giảm chiều dữ liệu bằng SVD
4. Đánh giá bằng Silhouette Score, Davies-Bouldin, Calinski-Harabasz

### Kết Quả Nổi Bật:

* Mô hình chia dữ liệu thành **2 cụm chính**:

  * Cụm 1: Phản hồi tích cực về đãi ngộ, phúc lợi, môi trường
  * Cụm 2: Góp ý về sếp, quy trình, đào tạo cần cải thiện

### Hàm:

* `generate_cluster_features()`
* `run_clustering_model()`
* `visualize_clusters()`

---

## 🛠️ Công Nghệ & Thư Viện

* Python, pandas, scikit-learn
* underthesea, pyvi, nltk
* transformers (ViSoBERT)
* matplotlib, seaborn, WordCloud
* joblib

---

## ▶️ Cách Chạy Dự Án

### Chạy trên Notebook:

1. Mount Google Drive chứa `Reviews.xlsx`
2. Cài dependencies: `pip install -r requirements.txt`
3. Chạy notebook: `project_final.ipynb`

### Chạy giao diện Web:

```bash
git clone https://github.com/mrkiss-it/SentimentAndClustering.git
cd SentimentAndClustering
pip install -r requirements.txt
streamlit run main.py
```

Mở trình duyệt: [http://localhost:8501](http://localhost:8501)

---

## 🔗 Tham Khảo

* Mã nguồn: [github.com/mrkiss-it/SentimentAndClustering](https://github.com/mrkiss-it/SentimentAndClustering)
* Ứng dụng demo: [sentimentandclustering-kiss.streamlit.app](https://sentimentandclustering-kiss.streamlit.app/)
