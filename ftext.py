import fasttext
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
import string
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


# Modeli yükle
model_path = "cc.en.300.bin"  # Bu yolu güncelleyin
model = fasttext.load_model(model_path)

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

def load_abstracts_from_line_json(file_path):
    abstracts = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            try:
                json_data = json.loads(line)
                original_title = json_data['title']
                original_abstract = json_data['abstract']
                name=json_data['name']
                title = preprocess_text(original_title)
                abstract = preprocess_text(original_abstract)
                abstracts.append((title, abstract, original_title, original_abstract,name))
            except json.JSONDecodeError:
                print("Bir satır JSON olarak ayrıştırılamadı")
            except KeyError:
                print("Abstract veya Title anahtarı bulunamadı")
    return abstracts


# Test ve valid dosyalarından abstract'ları yükle
abstracts_test = load_abstracts_from_line_json('test.json')
abstracts_valid = load_abstracts_from_line_json('valid.json')

# Tüm abstract'ları birleştir
all_abstracts = abstracts_test + abstracts_valid

def compute_abstract_vectors(abstracts):
    return [model.get_sentence_vector(abstract[1]) for abstract in abstracts]  # abstract[1] refers to the preprocessed abstract


abstract_vectors = compute_abstract_vectors(all_abstracts)

def recommend_articles_from_processed(ft_interests_vector, listread, viewed_vectors=[]):
    if listread is None:
        listread = []

    useful_articles_vectors = []
    for abstract in all_abstracts:
        if abstract[4] in listread:  # '4' is the index for 'name' in the tuple
            useful_articles_vectors.append(model.get_sentence_vector(abstract[1]))  # '1' is the index for preprocessed abstract

    all_vectors = [ft_interests_vector] + useful_articles_vectors + viewed_vectors

    if all_vectors:
        combined_interest_vector = np.mean(all_vectors, axis=0)
    else:
        combined_interest_vector = ft_interests_vector

    similarity_scores = cosine_similarity([combined_interest_vector], abstract_vectors)
    most_similar_article_indices = np.argsort(similarity_scores[0])[-5:][::-1]

    most_similar_articles = [(all_abstracts[index][2], all_abstracts[index][3], similarity_scores[0][index], all_abstracts[index][4]) for index in most_similar_article_indices]
    return most_similar_articles