# -*- coding: utf-8 -*-
print("🚀 project_final.py starting import")
# Thư viện chuẩn
import re
import warnings
import unicodedata
import html
from tqdm import tqdm
tqdm.pandas()
from langdetect import detect

# Xử lý dữ liệu và NLP
import pandas as pd
import numpy as np
from underthesea import word_tokenize as vi_tokenize
import nltk
from nltk.corpus import stopwords
# Tải bộ stopwords tiếng Anh
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')

from nltk.tokenize import word_tokenize as en_tokenize
from underthesea import pos_tag as vi_pos_tag
from nltk import pos_tag as en_pos_tag

# Trực quan hóa
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Machine Learning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score, davies_bouldin_score
)

# Mô hình học máy
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering

import yake
from summa import keywords as textrank_keywords

# Lưu mô hình
import joblib
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
import gdown
import os
from sentence_transformers import SentenceTransformer

if not os.path.exists("clustering/sentence_bert.pkl"):
    gdown.download("https://drive.google.com/uc?id=1H7_KROPikN6ru4lccn7H7b3Iacbw6-xU", "clustering/sentence_bert.pkl", quiet=False)

if not os.path.exists("sentiment/stacking.pkl"):
    gdown.download("https://drive.google.com/uc?id=1fK7ItKl5GcJjxaw3M9IAP6gDyuQXstUz", "sentiment/stacking.pkl", quiet=False)

embedding_model = joblib.load("clustering/sentence_bert.pkl")

num_cols = ['Salary & benefits', 'Training & learning', 'Culture & fun',
            'Office & workspace', 'Management cares about me']

# Đọc từ khóa tích cực
pos_words = set(Path("files/positive_words.txt").read_text("utf-8").splitlines())

# Đọc từ khóa tiêu cực
neg_words = set(Path("files/negative_words.txt").read_text("utf-8").splitlines())

print(f"> Loaded {len(pos_words)} positive, {len(neg_words)} negative keywords.")

# Đọc emoji tích cực
pos_emoji = set(Path("files/positive_emoji.txt").read_text("utf-8").splitlines())

# Đọc emoji tiêu cực
neg_emoji = set(Path("files/negative_emoji.txt").read_text("utf-8").splitlines())

print(f"Loaded {len(pos_emoji)} positive emojis, {len(neg_emoji)} negative emojis")


# Stopwords tiếng Việt cơ bản
review_stopwords_vi = {
    'thì', 'của', 'và', 'là', 'ở', 'về', 'ta', 'tôi', 'bạn', 'anh', 'chị', 'em',
    'mình', 'các', 'những', 'một', 'hai', 'ba', 'này', 'đó', 'kia', 'cho', 'từ',
    'với', 'trong', 'ngoài', 'trên', 'dưới', 'sau', 'trước', 'giữa', 'bên', 'cạnh',
    'rất', 'khá', 'hơi', 'quá', 'đã', 'đang', 'sẽ', 'vẫn', 'còn', 'chưa', 'không'
}

# Stopwords tiếng Anh cơ bản
review_stopwords_en = {
    'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
    'from', 'as', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has',
    'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
    'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they',
    'my', 'your', 'his', 'her', 'its', 'our', 'their', 'me', 'him', 'us', 'them'
}

# Từ nhiễu cụ thể
noise_words = {
    'cai', 'chua', 'ro', 'rang', 'notthing', 'thien', 'xe', 'dở', 'chỗ',
    'can', 'thing', 'process', 'need', 'improve' # chỉ loại nếu chúng xuất hiện độc lập
}

# Cấu hình hiển thị và tắt cảnh báo
plt.rcParams['font.family'] = 'DejaVu Sans'
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
warnings.filterwarnings('ignore')

def clean_text(text):
    # Loại bỏ các ký tự đặc biệt không cần thiết
    text = re.sub(r'[^\w\s\.,!?;:]', ' ', text)

    # Xóa URL
    text = re.sub(r'http\S+|www\.\S+', '', text)

    # Chuẩn hóa khoảng trắng
    text = re.sub(r'\s+', ' ', text)

    # Loại bỏ khoảng trắng đầu/cuối
    text = text.strip()

    return text

def normalize_repeated_characters(text):
    """
    Chuẩn hóa các từ có ký tự lặp liên tiếp
    Ví dụ: "lònggggg" -> "lòng", "thiệtttt" -> "thiệt"
    """
    # Xử lý ký tự Việt Nam (bao gồm dấu)
    # Thay thế 3+ ký tự liên tiếp bằng 1 ký tự
    text = re.sub(r'([aăâeêiouôơưyàáảãạằắẳẵặầấẩẫậèéẻẽẹềếểễệìíỉĩịòóỏõọồốổỗộờớởỡợùúủũụừứửữựỳýỷỹỵ])\1{2,}', r'\1', text, flags=re.IGNORECASE)

    # Xử lý consonant
    text = re.sub(r'([bcdfghjklmnpqrstvwxz])\1{2,}', r'\1', text, flags=re.IGNORECASE)

    return text

def normalize_punctuation(text):
    """Chuẩn hóa các dấu câu"""

    # Chuẩn hóa dấu chấm
    text = re.sub(r'\.{2,}', '.', text)

    # Chuẩn hóa dấu hỏi chấm
    text = re.sub(r'\?{2,}', '?', text)

    # Chuẩn hóa dấu cảm thán
    text = re.sub(r'!{2,}', '!', text)

    # Chuẩn hóa dấu phẩy
    text = re.sub(r',{2,}', ',', text)

    # Loại bỏ dấu nháy đơn (nếu cần)
    text = text.replace("'", "")

    # Chuẩn hóa khoảng trắng xung quanh dấu câu
    text = re.sub(r'\s*([.!?,:;])\s*', r'\1 ', text)
    text = re.sub(r'\s+', ' ', text)

    return text.strip()

def normalize_vietnamese(text):
    """Chuẩn hóa unicode tiếng Việt sử dụng unicodedata"""
    return unicodedata.normalize('NFC', text)

def process_special_chars(text):
    # Decode HTML entities
    text = html.unescape(text)

    # Xử lý emoji (tùy task)
    # Có thể giữ nguyên hoặc thay thế bằng text mô tả

    return text

def process_basic_text(text, max_length=256):
    print("process_basic_text...")
    # 1. Làm sạch cơ bản
    text = clean_text(text)

    # 2. Chuẩn hóa ký tự lặp
    text = normalize_repeated_characters(text)

    # 3. Chuẩn hóa dấu câu
    text = normalize_punctuation(text)

    # 4. Chuẩn hóa tiếng Việt
    text = normalize_vietnamese(text)

    # 5. Xử lý ký tự đặc biệt
    text = process_special_chars(text)

    return text

def calc_sentiment_features(text):
    toks = text.lower().strip()
    pos_w = sum(t in toks for t in pos_words)
    neg_w = sum(t in toks for t in neg_words)
    pos_e = sum(text.count(e) for e in pos_emoji)
    neg_e = sum(text.count(e) for e in neg_emoji)

    total_w = pos_w + neg_w
    total_e = pos_e + neg_e
    total_we = total_w + total_e

    ratio_words = (pos_w - neg_w) / total_w if total_w else 0
    ratio_emoji = (pos_e - neg_e) / total_e if total_e else 0
    ratio_all = (pos_w + pos_e - neg_w - neg_e) / total_we if total_we else 0

    return pos_w, neg_w, pos_e, neg_e, total_we, ratio_all

def detect_lang_safe(text):
    # Hàm detect language đơn giản, bạn có thể dùng langdetect hoặc rule riêng
    try:
        return detect(text)
    except:
        return ''

from collections import deque
def split_sentences_by_meaning(text, lang='vi'):
    """
    Tách câu theo dấu câu và một số liên từ chia ý.

    Args:
        text (str): Input text to split
        lang (str): Language code ('vi' for Vietnamese, 'en' for English)

    Returns:
        list: List of split sentences
    """
    # Danh sách các liên từ và từ nối phổ biến dùng để tách ý
    if lang == 'vi':
        split_keywords = [
            r'\bnhưng\b', r'\btuy nhiên\b', r'\btuy\b', r'\bmặc dù\b',
            r'\bdù\b', r'\bvì vậy\b', r'\bvì thế\b', r'\bdo đó\b',
            r'\bcũng\b', r'\bsong\b',
        ]
    elif lang == 'en':
        split_keywords = [
            r'\bbut\b', r'\bhowever\b', r'\bnevertheless\b', r'\bnonetheless\b',
            r'\balthough\b', r'\bthough\b', r'\beven though\b', r'\bwhile\b',
            r'\bwhereas\b', r'\btherefore\b', r'\bthus\b', r'\bhence\b',
            r'\bconsequently\b', r'\bmoreover\b', r'\bfurthermore\b', r'\bin addition\b',
            r'\byet\b', r'\bso\b', r'\bfor\b', r'\bnor\b',
            r'\botherwise\b', r'\bmeanwhile\b', r'\bon the other hand\b',
            r'\bin contrast\b', r'\bsimilarly\b', r'\blikewise\b'
        ]
    else:
        # Default to Vietnamese keywords
        split_keywords = [
            r'\bnhưng\b', r'\btuy nhiên\b', r'\btuy\b', r'\bmặc dù\b',
            r'\bdù\b', r'\bvì vậy\b', r'\bvì thế\b', r'\bdo đó\b',
            r'\bcũng\b', r'\bsong\b',
        ]

    # Bước 1: Chuẩn hóa văn bản
    text = text.strip()
    if not text:
        return []

    # Bước 2: Tách câu theo dấu ngắt (., !, ?)
    sentences = re.split(r'(?<=[.!?])\s+', text)

    final_sentences = []

    # Bước 3: Với mỗi câu, tiếp tục tách nếu có liên từ mang nhiều ý
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        # Ghép các keyword lại thành regex OR
        pattern = '|'.join(split_keywords)

        # Tách nếu có từ nối nhiều ý
        sub_sentences = re.split(pattern, sentence, flags=re.IGNORECASE)

        # Flatten the list of lists and strip each element
        for sub_sentence in sub_sentences:
            split_do_neu_result = recursive_split_sentences(sub_sentence, lang)
            final_sentences.extend([s.strip() for s in split_do_neu_result if s.strip()])

    return final_sentences

def recursive_split_sentences(text, lang='vi'):
    """
    Recursively split sentences based on patterns for the specified language.

    Args:
        text (str): Input text to split
        lang (str): Language code ('vi' for Vietnamese, 'en' for English)

    Returns:
        list: List of split sentences
    """
    # Define patterns for each language
    if lang == 'vi':
        patterns = [
            r'\bmỗi tội\b.*?\bdo\b.*?(?=,|\.|$)',
            r'\bdo\b.*?\bnên\b.*?(?=,|\.|$)',
            r'\bvì\b.*?\bnên\b.*?(?=,|\.|$)',
            r'\bmặc dù\b.*?\bnhưng\b.*?(?=,|\.|$)',
            r'\bnếu\b.*?\bthì\b.*?(?=,|\.|$)',
        ]
    elif lang == 'en':
        patterns = [
            r'\bif\b.*?\bthen\b.*?(?=,|\.|$)',
            r'\bwhen\b.*?\bthen\b.*?(?=,|\.|$)',
            r'\bsince\b.*?\btherefore\b.*?(?=,|\.|$)',
            r'\bbecause\b.*?\bso\b.*?(?=,|\.|$)',
            r'\bas\b.*?\bso\b.*?(?=,|\.|$)',
            r'\balthough\b.*?\byet\b.*?(?=,|\.|$)',
            r'\bthough\b.*?\bstill\b.*?(?=,|\.|$)',
            r'\bwhile\b.*?\bhowever\b.*?(?=,|\.|$)',
            r'\bunless\b.*?\botherwise\b.*?(?=,|\.|$)',
            r'\beven if\b.*?\bstill\b.*?(?=,|\.|$)',
        ]
    else:
        # Default to Vietnamese patterns
        patterns = [
            r'\bmỗi tội\b.*?\bdo\b.*?(?=,|\.|$)',
            r'\bdo\b.*?\bnên\b.*?(?=,|\.|$)',
            r'\bvì\b.*?\bnên\b.*?(?=,|\.|$)',
            r'\bmặc dù\b.*?\bnhưng\b.*?(?=,|\.|$)',
            r'\bnếu\b.*?\bthì\b.*?(?=,|\.|$)',
        ]

    queue = deque([text.strip()])
    results = []

    while queue:
        sentence = queue.popleft().strip(" ,.")

        matched = False
        for pattern in patterns:
            match = re.search(pattern, sentence, re.IGNORECASE)
            if match:
                matched = True
                # Phần trước mẫu
                before = sentence[:match.start()].strip(" ,.")
                # Phần trùng với mẫu
                middle = match.group().strip(" ,.")
                # Phần sau mẫu
                after = sentence[match.end():].strip(" ,.")

                if before:
                    queue.append(before)
                results.append(middle + ".")  # Đưa thẳng vào kết quả, không xử lý lại
                if after:
                    queue.append(after)
                break  # Chỉ dùng 1 pattern mỗi vòng

        if not matched:
            if sentence:
                results.append(sentence + ".")

    return results

# Hàm tiện ích để load tất cả dữ liệu
def load_processing_data():
    """Load tất cả dữ liệu cần thiết cho việc xử lý văn bản"""

    # LOAD EMOJICON
    try:
        with open('files/emojicon.txt', 'r', encoding="utf8") as file:
            emoji_lst = file.read().split('\n')
        emoji_dict = {}
        for line in emoji_lst:
            if '\t' in line and line.strip():  # Kiểm tra format và không rỗng
                line = normalize_vietnamese(line)  # Chuẩn hóa unicode
                parts = line.split('\t', 1)  # Chỉ split lần đầu
                if len(parts) == 2:
                    key, value = parts
                    emoji_dict[key] = str(value)
    except FileNotFoundError:
        print("Warning: emojicon.txt not found, using empty emoji dict")
        emoji_dict = {}

    # LOAD TEENCODE
    try:
        with open('files/teencode.txt', 'r', encoding="utf8") as file:
            teen_lst = file.read().split('\n')
        teen_dict = {}
        for line in teen_lst:
            if '\t' in line and line.strip():
                line = normalize_vietnamese(line)  # Chuẩn hóa unicode
                parts = line.split('\t', 1)
                if len(parts) == 2:
                    key, value = parts
                    teen_dict[key] = str(value)
    except FileNotFoundError:
        print("Warning: teencode.txt not found, using empty teen dict")
        teen_dict = {}

    #LOAD TRANSLATE ENGLISH -> VNMESE
    try:
        with open('files/english-vnmese.txt', 'r', encoding="utf8") as file:
            english_lst = file.read().split('\n')
        english_dict = {}
        for line in english_lst:
            if '\t' in line and line.strip():
                line = normalize_vietnamese(line)  # Chuẩn hóa unicode
                parts = line.split('\t', 1)
                if len(parts) == 2:
                    key, value = parts
                    english_dict[key] = str(value)
    except FileNotFoundError:
        print("Warning: teencode.txt not found, using empty teen dict")
        english_dict = {}

    # LOAD WRONG WORDS
    try:
        with open('files/wrong-word.txt', 'r', encoding="utf8") as file:
            wrong_lst = file.read().split('\n')
        wrong_lst = [normalize_vietnamese(word.strip()) for word in wrong_lst if word.strip()]  # Chuẩn hóa unicode
    except FileNotFoundError:
        print("Warning: wrong-word.txt not found, using empty wrong list")
        wrong_lst = []


    # LOAD STOPWORDS
    try:
        with open('files/vietnamese-stopwords.txt', 'r', encoding="utf8") as file:
            stopwords_vi = file.read().split('\n')
        stopwords_vi = [normalize_vietnamese(word.strip()) for word in stopwords_vi if word.strip()]  # Chuẩn hóa unicode
        # Loại bỏ các từ phủ định trong stopword
        negations_vi = {'không', 'chẳng', 'chả', 'đâu', 'chưa', 'đừng', 'khỏi'}
        stopwords_vi = [word for word in stopwords_vi if word not in negations_vi]
        # Loải bỏ các từ quan trọng
        keep_words = {'rất', 'không', 'nên', 'có', 'tốt', 'xấu', 'làm việc', 'áp lực', 'thoải mái', 'ổn', 'hài lòng', 'khó', 'tệ', 'cực kỳ', 'nhiều', 'ít', 'cao', 'thấp', 'khá'}
        stopwords_vi = [word for word in stopwords_vi if word not in keep_words]
    except FileNotFoundError:
        print("Warning: vietnamese-stopwords.txt not found, using empty stopwords list")
        stopwords_vi = []

    # Lấy bộ stopwords tiếng Anh
    stopwords_en = set(stopwords.words('english'))

    # Loại bỏ từ phủ định khỏi bộ stopwords
    negations = {'not', 'no', 'nor', 'don', 'didn', 'doesn', "don't", "didn't", "doesn't"}
    stopwords_en = [word for word in stopwords_en if word not in negations]
    # Loại bỏ các từ quan trọng
    keep_words = {'not', 'very', 'no', 'never', 'good', 'bad', 'great', 'poor', 'excellent', 'happy', 'sad', 'stressful', 'comfortable', 'satisfied', 'unhappy', 'awful', 'amazing'}
    stopwords_en = [word for word in stopwords_en if word not in keep_words]

    return emoji_dict, teen_dict, english_dict, wrong_lst, stopwords_vi, stopwords_en

emoji_dict, teen_dict, english_dict, wrong_lst, stopwords_vi, stopwords_en = load_processing_data()
# Kết hợp cuối cùng
all_stopwords = set(stopwords_vi) | set(stopwords_en) | review_stopwords_vi | review_stopwords_en | noise_words


def process_split_text(split_text, lang='vi'):
    """
    Xử lý nhiều câu cùng lúc

    """
    results = []
    for text in split_text:
        if not text or not text.strip():
            continue

        results.extend(process_text(text, lang))
    # loại bỏ câu trùng
    results = list(set(results))
    # nối thành 1 chuỗi
    return '. '.join(results)

def process_text(text, lang='vi'):
    if not text or not text.strip():
        return ""

    # Bước 1: Thay emoji thành từ tương ứng
    text = replace_emoji(text, emoji_dict)

    # Bước 2: Chuẩn hóa teencode
    text = normalize_text(text, teen_dict)

    if lang == 'vi':
        tokens = vi_tokenize(text, format="text")
        tagged = vi_pos_tag(tokens)
        results = process_tagged_sentence(tagged).split('.')
        results = [s.strip() for s in results if s.strip()]
    else:
        tokens = en_tokenize(text)
        tagged = en_pos_tag(tokens)
        results = process_tagged_sentence(tagged).split('.')
        results = [s.strip() for s in results if s.strip()]

    return results

def normalize_text(text, dict_map):
    # Thay thế từ theo dict_map (teencode, sai chính tả, emoji...)
    pattern = re.compile(r'\b(' + '|'.join(re.escape(key) for key in dict_map.keys()) + r')\b', flags=re.IGNORECASE)
    def replace_func(match):
        key = match.group(0).lower()
        return dict_map.get(key, key)
    return pattern.sub(replace_func, text)

def replace_emoji(text, emoji_dict):
    # Thay emoji từng ký tự
    for emo, rep in emoji_dict.items():
        text = text.replace(emo, ' ' + rep + ' ')
    return text

def process_tagged_sentence(tagged_words, language='vi'):
    """
    Xử lý câu đã được gán nhãn từ loại để tạo câu có nghĩa với phân cụm ý nghĩa
    Hỗ trợ cả tiếng Việt và tiếng Anh

    Args:
        tagged_words: Danh sách các tuple (từ, nhãn_từ_loại)
        language: 'vi' cho tiếng Việt, 'en' cho tiếng Anh

    Returns:
        str: Câu đã được xử lý, phân thành các cụm ý nghĩa
    """

    # Định nghĩa các từ loại cần giữ lại (có nghĩa) cho cả tiếng Việt và tiếng Anh
    meaningful_tags = {
        'N', 'NN', 'NNS', 'NNP', 'NNPS',  # Danh từ (Noun)
        'V', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ',  # Động từ (Verb)
        'A', 'JJ', 'JJR', 'JJS',  # Tính từ (Adjective)
    }

    # Định nghĩa các từ cụ thể cần loại bỏ theo ngôn ngữ
    if language == 'vi':
        stop_words = {
            'có_thể', 'ở', 'từ', 'vì', 'mỗi', 'của', 'với', 'theo', 'trong', 'trên', 'dưới'
        }
        # Trạng từ quan trọng cần giữ
        important_adverbs = {'không', 'khá', 'rất', 'tương_đối', 'khá_là', 'đã', 'sẽ', 'đang'}
        # Từ quan trọng cần giữ bất kể từ loại
        important_words = {'nên', 'và', 'hoặc', 'cũng', 'thì', 'là', 'được'}
        # Từ loại dấu câu và giới từ cần bỏ qua
        skip_tags = {'CH', 'E', 'C'}
    else:  # English
        stop_words = {
            'of', 'in', 'on', 'at', 'by', 'for', 'with', 'from', 'to', 'about',
            'into', 'through', 'during', 'before', 'after', 'above', 'below', 'between'
        }
        # Trạng từ quan trọng cần giữ
        important_adverbs = {'not', 'very', 'quite', 'really', 'too', 'so', 'already', 'still', 'just'}
        # Từ quan trọng cần giữ bất kể từ loại
        important_words = {'and', 'or', 'but', 'should', 'must', 'can', 'will', 'would', 'could', 'is', 'are', 'was', 'were', 'be', 'been', 'being'}
        # Từ loại dấu câu và giới từ cần bỏ qua
        skip_tags = {'.', ',', ':', ';', '!', '?', 'IN', 'TO', 'CC'}

    meaningful_words = []
    current_chunk = []
    check_tags = set()

    for i, (word, tag) in enumerate(tagged_words):
        # Bỏ qua các từ trong danh sách stop words
        if word.lower() in stop_words:
            continue

        # Xử lý dấu phẩy - kết thúc cụm hiện tại
        if word == ',' and current_chunk:
            if len(check_tags) == 1 and meaningful_words:
                last_text = meaningful_words.pop()
                meaningful_words.append(last_text + ', ' + ' '.join(current_chunk))
            else:
                meaningful_words.append(' '.join(current_chunk))
            current_chunk = []
            check_tags = set()
            continue

        # Bỏ qua dấu chấm cuối câu
        if word == '.' and i == len(tagged_words) - 1:
            continue

        # Giữ lại từ quan trọng bất kể từ loại
        if word.lower() in important_words:
            current_chunk.append(word)
            check_tags.add(tag)
            continue

        # Giữ lại các từ có nghĩa
        if tag in meaningful_tags:
            # Kiểm tra trạng từ quan trọng
            if tag in {'R', 'RB', 'RBR', 'RBS'} and word.lower() not in important_adverbs:
                continue
            current_chunk.append(word)
            check_tags.add(tag)
        elif tag not in skip_tags:  # Giữ lại các tag khác nếu không phải dấu câu, giới từ, liên từ
            current_chunk.append(word)
            check_tags.add(tag)

    # Thêm cụm cuối cùng nếu còn
    if current_chunk:
        if len(check_tags) == 1 and meaningful_words:
              last_text = meaningful_words.pop()
              meaningful_words.append(last_text + ', ' + ' '.join(current_chunk))
        else:
            meaningful_words.append(' '.join(current_chunk))

    # Hàm viết hoa chữ cái đầu
    def capitalize_first_word(text):
        if not text:
            return text
        words = text.split()
        if words and len(words[0]) > 0:
            first_word = words[0]
            words[0] = first_word[0].upper() + first_word[1:]
        return ' '.join(words)

    # Tạo câu với định dạng mong muốn và viết hoa
    if len(meaningful_words) == 1:
        return capitalize_first_word(meaningful_words[0]) + "."
    elif len(meaningful_words) == 2:
        return (capitalize_first_word(meaningful_words[0]) + ". " +
                capitalize_first_word(meaningful_words[1]) + ".")
    elif len(meaningful_words) >= 3:
        # Nối 2 cụm đầu bằng dấu phẩy, các cụm sau bằng dấu chấm
        result = capitalize_first_word(meaningful_words[0]) + ", " + capitalize_first_word(meaningful_words[1]) + ". "
        for chunk in meaningful_words[2:]:
            result += capitalize_first_word(chunk) + ". "
        return result.rstrip()

    return ""

def evaluate_multiple_models(X, k_range=range(2, 9), dbscan_eps=0.5, dbscan_min_samples=5):
    """So sánh KMeans, Agglomerative và DBSCAN với các chỉ số đánh giá"""
    results = []

    for k in k_range:
        # KMeans
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels_kmeans = kmeans.fit_predict(X)
        sil_kmeans = silhouette_score(X, labels_kmeans)
        db_kmeans = davies_bouldin_score(X, labels_kmeans)
        ch_kmeans = calinski_harabasz_score(X, labels_kmeans)
        results.append({
            'Model': 'KMeans', 'k': k,
            'Silhouette': sil_kmeans,
            'DaviesBouldin': db_kmeans,
            'CalinskiHarabasz': ch_kmeans
        })

        # Agglomerative
        agg = AgglomerativeClustering(n_clusters=k)
        labels_agg = agg.fit_predict(X)
        sil_agg = silhouette_score(X, labels_agg)
        db_agg = davies_bouldin_score(X, labels_agg)
        ch_agg = calinski_harabasz_score(X, labels_agg)
        results.append({
            'Model': 'Agglomerative', 'k': k,
            'Silhouette': sil_agg,
            'DaviesBouldin': db_agg,
            'CalinskiHarabasz': ch_agg
        })

    # DBSCAN - chỉ chạy 1 lần vì không có k
    dbscan = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
    labels_dbscan = dbscan.fit_predict(X)
    if len(set(labels_dbscan)) > 1 and -1 not in set(labels_dbscan):
        sil_db = silhouette_score(X, labels_dbscan)
        db_db = davies_bouldin_score(X, labels_dbscan)
        ch_db = calinski_harabasz_score(X, labels_dbscan)
        results.append({
            'Model': 'DBSCAN', 'k': len(set(labels_dbscan)),
            'Silhouette': sil_db,
            'DaviesBouldin': db_db,
            'CalinskiHarabasz': ch_db
        })

    return pd.DataFrame(results)

# 7. Gán clustering theo model tốt nhất
def get_cluster_labels(model_name, X, k):
    if model_name == 'KMeans':
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
    elif model_name == 'Agglomerative':
        model = AgglomerativeClustering(n_clusters=k)
    else:
        raise ValueError("Model không hỗ trợ!")

    pred = model.fit_predict(X)

    return model, pred

def evaluate_clustering(X, labels, name):
    """Đánh giá chất lượng clustering"""
    silhouette = silhouette_score(X, labels)
    davies_bouldin = davies_bouldin_score(X, labels)
    calinski_harabasz = calinski_harabasz_score(X, labels)

    print(f"\n{name}:")
    print(f"  Silhouette Score: {silhouette:.3f}")
    print(f"  Davies-Bouldin Index: {davies_bouldin:.3f}")
    print(f"  Calinski-Harabasz Index: {calinski_harabasz:.3f}")

    return silhouette, davies_bouldin, calinski_harabasz

def analyze_cluster_content(df, cluster_col, text_col, n_samples=3):
    """Phân tích nội dung của từng cluster"""
    print(f"\n--- Phân tích {cluster_col} ---")

    for cluster_id in sorted(df[cluster_col].unique()):
        cluster_data = df[df[cluster_col] == cluster_id]
        print(f"\nCluster {cluster_id} (n={len(cluster_data)}):")

        # Hiển thị một số mẫu
        samples = cluster_data[text_col].head(n_samples)
        for i, sample in enumerate(samples, 1):
            print(f"  {i}. {sample[:100]}...")

        # Thống kê rating
        print(f"  Trung bình rating:")
        for col in num_cols:
            if col in cluster_data.columns:
                avg_rating = cluster_data[col].mean()
                print(f"    {col}: {avg_rating:.2f}")

def plot_top_keywords_tfidf(df, cluster_col, text_col, top_n=20):
    for cluster_id in sorted(df[cluster_col].unique()):
        text = df[df[cluster_col] == cluster_id][text_col].dropna().tolist()
        vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(6, 6), min_df=5, max_df=0.7, lowercase=True, analyzer='word', sublinear_tf=True)
        tfidf_matrix = vectorizer.fit_transform(text)
        tfidf_mean = tfidf_matrix.mean(axis=0).A1
        keywords = pd.Series(tfidf_mean, index=vectorizer.get_feature_names_out()).sort_values(ascending=False)[:top_n]

        print(f"\nCluster {cluster_id} top keywords:")
        print(keywords)

        # WordCloud
        wc = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(keywords)
        plt.figure(figsize=(10, 5))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        plt.title(f"WordCloud - Cluster {cluster_id}")
        plt.show()

def extract_textrank_keywords(text, top_n=15):
    """
    Hàm riêng để extract TextRank keywords với nhiều chiến lược
    """
    all_keywords = []

    try:
        # Chiến lược 1: Lấy cụm từ (split=False)
        phrases = textrank_keywords.keywords(text, words=min(top_n, 10), split=False)
        if phrases:
            phrase_list = [p.strip() for p in phrases.split('\n') if p.strip()]
            all_keywords.extend(phrase_list)
    except:
        pass

    try:
        # Chiến lược 2: Lấy từ đơn (split=True)
        words = textrank_keywords.keywords(text, words=min(top_n, 10), split=True)
        if words:
            all_keywords.extend(words)
    except:
        pass

    try:
        # Chiến lược 3: Tạo n-gram từ top words
        if len(all_keywords) > 3:
            # Tạo bigram và trigram từ các từ quan trọng
            words_for_ngram = [w for w in all_keywords if len(w.split()) == 1][:8]
            text_words = text.lower().split()

            # Tìm bigram
            for i in range(len(text_words) - 1):
                bigram = f"{text_words[i]} {text_words[i+1]}"
                if any(word in bigram for word in words_for_ngram):
                    all_keywords.append(bigram)

            # Tìm trigram
            for i in range(len(text_words) - 2):
                trigram = f"{text_words[i]} {text_words[i+1]} {text_words[i+2]}"
                if any(word in trigram for word in words_for_ngram):
                    all_keywords.append(trigram)
    except:
        pass

    return all_keywords

def is_valid_keyword(kw):
    """Cải thiện validation cho keyword"""
    kw_clean = kw.strip()

    # Kiểm tra độ dài
    if len(kw_clean) <= 2: return False

    # Kiểm tra stopwords
    if kw_clean.lower() in all_stopwords: return False

    # Kiểm tra ký tự đặc biệt
    if re.match(r'^[\W_]+$', kw_clean): return False

    # Kiểm tra số
    if kw_clean.isdigit(): return False

    # Kiểm tra từ đơn quá ngắn và phổ biến
    single_word_blacklist = {'work', 'good', 'great', 'nice', 'bad', 'ok', 'yes', 'no',
                            'công', 'việc', 'tốt', 'được', 'có', 'là', 'và', 'không', 'cho'}
    if len(kw_clean.split()) == 1 and kw_clean.lower() in single_word_blacklist:
        return False

    # Kiểm tra cụm từ kết thúc bằng dấu câu
    if kw_clean.endswith(('.', ':', ',', ';', '!', '?')):
        return False

    # Kiểm tra cụm từ có ít nhất 1 từ có nghĩa (>= 3 ký tự)
    meaningful_words = [w for w in kw_clean.split() if len(w) >= 3]
    if len(meaningful_words) == 0:
        return False

    # Kiểm tra tỷ lệ ký tự đặc biệt
    special_chars = sum(1 for c in kw_clean if not c.isalnum() and c != ' ')
    if special_chars > len(kw_clean) * 0.3:  # >30% ký tự đặc biệt
        return False

    return True

def clean_keyword(kw):
    """Làm sạch keyword"""
    # Loại bỏ dấu câu ở đầu và cuối
    kw_clean = re.sub(r'^[\W_]+|[\W_]+$', '', kw.strip())

    # Loại bỏ ký tự đặc biệt thừa
    kw_clean = re.sub(r'[^\w\s]', ' ', kw_clean)

    # Loại bỏ khoảng trắng thừa
    kw_clean = ' '.join(kw_clean.split())

    return kw_clean

def label_cluster_with_all_methods(df, cluster_col, text_col, top_n=10, display=True, batch_size=3):
    results = {}
    cluster_ids = sorted(df[cluster_col].unique())

    print(f"Đang xử lý {len(cluster_ids)} clusters của {cluster_col}...")

    for i, cluster_id in enumerate(cluster_ids):
        print(f"Xử lý cluster {cluster_id} ({i+1}/{len(cluster_ids)})")

        # Lọc data cho cluster hiện tại
        cluster_data = df[df[cluster_col] == cluster_id].copy()
        cluster_data[text_col] = cluster_data.progress_apply(lambda x: split_sentences_by_meaning(x[text_col], x['lang']), axis=1)
        cluster_data[text_col] = cluster_data.progress_apply(lambda x: process_split_text(x[text_col], x['lang']), axis=1)


        # Xử lý từng ngôn ngữ
        texts_vi = cluster_data[cluster_data['lang'] == 'vi'][text_col].dropna().tolist()
        texts_en = cluster_data[cluster_data['lang'] == 'en'][text_col].dropna().tolist()

        # Giới hạn số lượng text để tiết kiệm RAM
        texts_vi = texts_vi[:100]  # Chỉ lấy 50 text đầu tiên
        texts_en = texts_en[:50]

        joined_text_vi = " ".join(texts_vi)
        joined_text_en = " ".join(texts_en)

        joined_text = joined_text_vi + " " + joined_text_en

        # Bỏ qua cluster nếu text quá ngắn
        if len(joined_text.strip()) < 50:
            results[cluster_id] = ["insufficient_data"]
            continue

        try:
            # YAKE
            kw_yake_vi, kw_yake_en = [], []
            if joined_text_vi.strip():
                kw_yake_vi = yake.KeywordExtractor(lan="vi", n=3, top=min(top_n, 15)).extract_keywords(joined_text_vi)
            if joined_text_en.strip():
                kw_yake_en = yake.KeywordExtractor(lan="en", n=3, top=min(top_n, 15)).extract_keywords(joined_text_en)

            # Clean và filter YAKE keywords
            yake_keywords = []
            for kw, score in kw_yake_vi + kw_yake_en:
                cleaned = clean_keyword(kw)
                if cleaned and is_valid_keyword(cleaned):
                    yake_keywords.append(cleaned)
            set_yake = set(yake_keywords)

            # TextRank
            set_textrank = set()
            try:
                if len(joined_text.strip()) > 100:
                    all_textrank = extract_textrank_keywords(joined_text, top_n)
                    set_textrank = set([kw for kw in all_textrank if is_valid_keyword(kw)])
                    # Clean và filter TextRank keywords
                    textrank_keywords = []
                    for kw in all_textrank:
                        cleaned = clean_keyword(kw)
                        if cleaned and is_valid_keyword(cleaned):
                            textrank_keywords.append(cleaned)
                    set_textrank = set(textrank_keywords)
            except Exception as e:
                print(f"TextRank failed for cluster {cluster_id}: {e}")
                set_textrank = set()

            # Tổng hợp và voting
            all_keywords = list(set_yake | set_textrank)
            voting_score = {}
            for kw in all_keywords:
                voting_score[kw] = (
                    (kw in set_yake) +
                    (kw in set_textrank)
                )

            # Ưu tiên cụm ≥2 từ & voting >= 2
            final_labels = sorted(
                [kw for kw in all_keywords if len(kw.split()) >= 2 and voting_score[kw] >= 2],
                key=lambda x: (-voting_score[x], joined_text.find(x))
            )[:top_n]

            if not final_labels:  # fallback nếu không có từ khóa mạnh
                final_labels = sorted(
                    [kw for kw in all_keywords if is_valid_keyword(kw)],
                    key=lambda x: (-voting_score.get(x, 0), -len(x), joined_text.find(x))
                )[:top_n]

            results[cluster_id] = final_labels

            if display:
                print(f"\n📌 Cluster {cluster_id} - Label gợi ý: {final_labels}")
                print(f"→ YAKE: {list(set_yake)[:3]}")
                print(f"→ TextRank: {list(set_textrank)[:3]}")

        except Exception as e:
            print(f"Lỗi khi xử lý cluster {cluster_id}: {e}")
            results[cluster_id] = ["error_processing"]

    return results

def get_top_tfidf_keywords_by_cluster(df, cluster_col, text_col, top_n=10):
    results = {}
    # Gộp tất cả văn bản theo cluster
    grouped_text = df.groupby(cluster_col)[text_col + '_advance'].apply(lambda texts: ' '.join(texts.dropna()))

    vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(5, 8), min_df=2, max_df=1.0, lowercase=True, analyzer='word', sublinear_tf=True, stop_words=None)

    tfidf_matrix = vectorizer.fit_transform(grouped_text)
    feature_names = vectorizer.get_feature_names_out()

    for i, cluster in enumerate(grouped_text.index):
        row = tfidf_matrix[i].toarray().flatten()
        top_indices = row.argsort()[::-1][:top_n]
        keywords = [feature_names[j] for j in top_indices if row[j] > 0]
        results[cluster] = keywords

    return results

def get_best_label_by_embedding(reduced_data, cluster_labels, keyword_dict):
    cluster_names = {}
    for cluster_id in sorted(set(cluster_labels)):
        idx = np.where(cluster_labels == cluster_id)[0]
        centroid = reduced_data[idx].mean(axis=0).reshape(1, -1)

        keywords = keyword_dict.get(cluster_id, [])
        keyword_embeds = embedding_model.encode(keywords, convert_to_numpy=True)

        sims = cosine_similarity(keyword_embeds, centroid).flatten()
        best_keyword = keywords[np.argmax(sims)] if len(sims) > 0 else "N/A"

        cluster_names[cluster_id] = best_keyword

        print(f"📌 Cluster {cluster_id} → Nhãn chọn theo embedding: {best_keyword}")
    return cluster_names

def rank_keywords_by_similarity(reduced_data, cluster_labels, keyword_dict):
    cluster_rankings = {}

    for cluster_id in sorted(set(cluster_labels)):
        idx = np.where(cluster_labels == cluster_id)[0]
        centroid = reduced_data[idx].mean(axis=0).reshape(1, -1)

        keywords = keyword_dict.get(cluster_id, [])
        if not keywords:
            cluster_rankings[cluster_id] = []
            continue

        keyword_embeds = embedding_model.encode(keywords, convert_to_numpy=True)
        sims = cosine_similarity(keyword_embeds, centroid).flatten()

        ranked = sorted(zip(keywords, sims), key=lambda x: -x[1])
        cluster_rankings[cluster_id] = ranked

        print(f"\n📌 Cluster {cluster_id} - Ranking:")
        for kw, sim in ranked:
            print(f"{kw:30s} → similarity: {sim:.4f}")

    return cluster_rankings

def get_representative_samples(reduced_data, cluster_labels, df_text, top_n=5):
    cluster_samples = {}

    for cluster_id in sorted(set(cluster_labels)):
        idx = np.where(cluster_labels == cluster_id)[0]
        centroid = reduced_data[idx].mean(axis=0).reshape(1, -1)
        cluster_vecs = reduced_data[idx]
        sims = cosine_similarity(cluster_vecs, centroid).flatten()

        top_indices = idx[np.argsort(-sims)[:top_n]]
        samples = df_text.iloc[top_indices].tolist()

        print(f"\n📌 Top {top_n} reviews gần trung tâm Cluster {cluster_id}:")
        for i, s in enumerate(samples, 1):
            print(f"{i}. {s[:150]}...")  # In 150 ký tự đầu
        cluster_samples[cluster_id] = samples

    return cluster_samples

def summarize_clusters(cluster_labels, liked_reduced, keyword_rankings, df_text):
    summary = []
    cluster_ids = sorted(set(cluster_labels))

    for cluster_id in cluster_ids:
        keywords = [kw for kw, _ in keyword_rankings.get(cluster_id, [])[:5]]
        size = sum(cluster_labels == cluster_id)

        # Trích 1-2 câu đại diện
        idx = np.where(cluster_labels == cluster_id)[0]
        centroid = liked_reduced[idx].mean(axis=0).reshape(1, -1)
        cluster_vecs = liked_reduced[idx]
        sims = cosine_similarity(cluster_vecs, centroid).flatten()
        top_indices = idx[np.argsort(-sims)[:2]]
        sample_preview = [df_text.iloc[i][:100] + "..." for i in top_indices]

        summary.append({
            "Cluster": cluster_id,
            "Top Keywords": ", ".join(keywords),
            "Samples": "\n".join(sample_preview),
            "Size": size
        })

    return pd.DataFrame(summary)

def show_examples(df, company, liked_c=None, suggested_c=None, n=3):
    filtered = df[df['Company Name'] == company]
    if liked_c is not None:
        filtered = filtered[filtered['liked_cluster'] == liked_c]
    if suggested_c is not None:
        filtered = filtered[filtered['suggested_cluster'] == suggested_c]

    display_cols = ['Company Name', 'What I liked_procced', 'Suggestions for improvement_procced', 'Rating']
    return filtered[display_cols].head(n)

def find_contradictions(liked_counts, suggested_counts, company):
    liked = liked_counts.loc[company] if company in liked_counts.index else None
    suggested = suggested_counts.loc[company] if company in suggested_counts.index else None

    print(f"\n📌 Company: {company}")
    print(f"Liked Clusters: \n{liked}")
    print(f"Suggested Clusters: \n{suggested}")

def get_key_words(data, top_n=20, min_score=0.05, ngram_ranges=[(2,4), (1, 2)]):
    text = " ".join(data)  # Gộp danh sách thành chuỗi

    if text.strip() == '':
        return pd.Series(dtype=float)

    if isinstance(text, str):
        text = [text]
    texts = [t.strip() for t in text if t.strip()]

    for ngram in ngram_ranges:
        print(f"\nThử ngram_range={ngram}:")
        vectorizer = TfidfVectorizer(ngram_range=ngram, stop_words=None)
        try:
            tfidf_matrix = vectorizer.fit_transform(texts)
            if tfidf_matrix.shape[1] == 0:
                print("⚠️ Không có từ nào phù hợp.")
                continue

            feature_names = vectorizer.get_feature_names_out()
            tfidf_mean = tfidf_matrix.mean(axis=0).A1

            top_features = pd.Series(tfidf_mean, index=feature_names).sort_values(ascending=False)

            # ✅ Lọc từ khóa theo ngưỡng điểm
            filtered = top_features[top_features >= min_score][:top_n]
            top_features.index = top_features.index.astype(str)

            print(f"Số từ vựng: {len(feature_names)}")
            print(f"Top {top_n} từ khóa (lọc theo min_score={min_score}):\n{filtered}")

            return filtered
        except Exception as e:
            print(f"Lỗi với ngram_range={ngram}: {e}")

    return pd.Series(dtype=float)

def check_wordcloud(keywords, col_name):
    """Tạo WordCloud từ dữ liệu text"""
    if keywords.empty:
        print("⚠️ Không có từ khóa để tạo WordCloud.")
        return None

    wc = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(keywords)
    # Tạo figure và vẽ WordCloud
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis("off")
    ax.set_title("WordCloud của " + col_name, fontsize=16, fontweight='bold', pad=20)

    return fig

