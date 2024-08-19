import json
from collections import Counter
def load_data(filename):
    keywords = []
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            try:
                article = json.loads(line.strip())  # Her bir satırı ayrı ayrı JSON objesi olarak yükle
                if 'keywords' in article:
                    # Anahtar kelimeleri ';' ile ayır ve boşlukları temizle
                    split_keywords = article['keywords'].split(';')
                    split_keywords = [kw.strip() for kw in split_keywords if kw.strip()]
                    keywords.extend(split_keywords)  # Anahtar kelimeleri listeye ekle
            except json.JSONDecodeError as e:
                print(f"JSON decode hatası: {str(e)} Satır: {line}")
                continue
    return keywords
def count_keywords(filenames):
    all_keywords = []
    for filename in filenames:
        keywords = load_data(filename)
        all_keywords.extend(keywords)

    keyword_frequency = Counter(all_keywords)  # Kelimelerin frekansını hesapla
    return keyword_frequency

# Dosya isimlerini bir listeye koy
filenames = ['test.json', 'valid.json']
# Anahtar kelimelerin frekanslarını hesapla ve yazdır
keyword_frequency = count_keywords(filenames)
for keyword, frequency in keyword_frequency.items():
    print(f"{keyword}: {frequency}")


def get_top_keywords(filenames, top_n=10):
    keyword_frequency = count_keywords(filenames)
    # En sık kullanılan anahtar kelimeleri döndür
    return keyword_frequency.most_common(top_n)
