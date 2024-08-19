from nltk import WordNetLemmatizer
from transformers import AutoTokenizer, AutoModel
import torch
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
nltk.download('wordnet')

nltk.download('punkt')
nltk.download('stopwords')

# SciBERT modelini ve tokenizer'ı yükle
tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')

stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Noktalama işaretlerini kaldır
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Küçük harfe çevir
    text = text.lower()
    # Tokenize et
    tokens = word_tokenize(text)
    # Durak sözcükleri kaldır
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if token not in stop_words]
    # Stemming
    lematizer=WordNetLemmatizer()
    stemmed_tokens = [lematizer.lemmatize(token) for token in filtered_tokens]
    # Temizlenmiş metni döndür
    return ' '.join(stemmed_tokens)

def get_sentence_vector(sentence):
    # Ön işlemeden geçirilen cümleyi tokenize et ve SciBERT ile vektör hesapla
    inputs = tokenizer(sentence, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    # Cümle vektörünü hesapla
    return outputs.last_hidden_state.mean(dim=1).squeeze()
def load_processed_abstracts(file_path):
    processed_abstracts = []
    with open(file_path, 'r', encoding='utf-8') as file:
        json_data = json.load(file)
        for item in json_data:
            title = item['title']
            abstract=item['abstract']
            vector = np.array(item['abstract_vector'])
            name=item['name']
            processed_abstracts.append((title,abstract, vector,name))
    return processed_abstracts

# Önceden işlenmiş vektör dosyalarını yükle
processed_test_abstracts = load_processed_abstracts('processed_test.json')
processed_valid_abstracts = load_processed_abstracts('processed_valid.json')

# Tüm işlenmiş abstract vektörlerini birleştir
all_processed_abstracts = processed_test_abstracts + processed_valid_abstracts

def recommend_articles_from_processed(interest_vector, listread):
    # listread None ise boş liste atayarak hata önle
    if listread is None:
        listread = []

    # Load useful articles' vectors based on 'listread' which contains names of useful articles
    useful_articles_vectors = []
    for abstract in all_processed_abstracts:
        if abstract[3] in listread:  # '3' is the index for 'name' in the tuple
            useful_articles_vectors.append(abstract[2])  # '2' is the index for vector in the tuple

    if useful_articles_vectors:
        # Calculate the average vector of useful articles
        useful_articles_mean_vector = np.mean(useful_articles_vectors, axis=0)
        # Combine this mean vector with the original interests vector
        combined_interest_vector = interest_vector + torch.tensor(useful_articles_mean_vector, dtype=torch.float32)
    else:
        # If no useful article vectors, use the original interests vector
        combined_interest_vector = interest_vector

    # Calculate cosine similarities with the updated interest vector
    abstract_vectors = [abstract[2] for abstract in all_processed_abstracts]
    similarity_scores = cosine_similarity([combined_interest_vector], abstract_vectors)
    most_similar_article_indices = np.argsort(similarity_scores[0])[-5:][::-1]

    # Return the most similar articles
    most_similar_articles = [(all_processed_abstracts[index][0], all_processed_abstracts[index][1], similarity_scores[0][index], all_processed_abstracts[index][3]) for index in most_similar_article_indices]
    return most_similar_articles