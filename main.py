import fasttext
import torch
from flask import Flask, render_template, request, redirect, url_for, session, flash
import pymongo
from pymongo import MongoClient
from werkzeug.security import generate_password_hash, check_password_hash
import json
import nltk
import numpy as np
from bson.objectid import ObjectId

from interests import get_top_keywords
from scibert import preprocess_text,  recommend_articles_from_processed,get_sentence_vector
from ftext import recommend_articles_from_processed as recommend_articles_ft

app = Flask(__name__)
app.secret_key = 'your_secret_key'

client = MongoClient("mongodb://localhost:27017/")
db = client["mydatabase"]
users_collection = db["users"]
articles_collection = db["articles"]

JSON_FILES = ['data/test.json', 'data/valid.json']

@app.route('/')
def home():
    return render_template('home.html')

model_path = "cc.en.300.bin"
model = fasttext.load_model(model_path)

def compute_interests_vector_scibert(interests):
    preprocessed_interests = [preprocess_text(interest) for interest in interests]
    interest_vectors = [get_sentence_vector(interest) for interest in preprocessed_interests]
    average_vector = torch.mean(torch.stack(interest_vectors), dim=0)
    return average_vector.numpy().tolist()

def compute_interests_vector_fasttext(interests):
    preprocessed_interests = [preprocess_text(interest) for interest in interests]
    interest_vectors = [model.get_sentence_vector(interest) for interest in preprocessed_interests]
    average_vector = np.mean(interest_vectors, axis=0)
    return average_vector.tolist()

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'GET':
        top_keywords = get_top_keywords(['data/test.json', 'data/valid.json'], 20)
        return render_template('signup.html', top_keywords=top_keywords)
    elif request.method == 'POST':
        interests = request.form.getlist('interests')
        if interests:
            interests_vector_scibert = compute_interests_vector_scibert(interests)
            interests_vector_fasttext = compute_interests_vector_fasttext(interests)
            user_data = {
                "email": request.form['email'],
                "password": generate_password_hash(request.form['password']),
                "name": request.form['name'],
                "surname": request.form['surname'],
                "age": request.form['age'],
                "gender": request.form['gender'],
                "interests": interests,
                "interests_vector_scibert": interests_vector_scibert,
                "interests_vector_fasttext": interests_vector_fasttext,
            }
            users_collection.insert_one(user_data)
            return redirect(url_for('login'))
        else:
            return "Lütfen en az bir ilgi alanı seçiniz.", 400
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():

    if request.method == 'GET':
        return render_template('login.html')
    elif request.method == 'POST':
        user = users_collection.find_one({"email": request.form['email']})
        if user and check_password_hash(user['password'], request.form['password']):
            session['user_id'] = str(user['_id'])
            session['user_email'] = user['email']
            session["sbertcounter"]=0
            session["truecountsb"]=0
            return redirect(url_for('article'))
        else:
            return "Email veya şifre hatalı!"

@app.route('/profile', methods=['GET', 'POST'])
def profile():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    user = users_collection.find_one({"_id": ObjectId(session['user_id'])})
    all_interests = get_top_keywords(['data/test.json', 'data/valid.json'], 20)

    if request.method == 'GET':
        user_interests = user.get('interests', [])
        return render_template('profile.html', user=user, all_interests=all_interests, user_interests=user_interests)
    elif request.method == 'POST':
        updated_interests = request.form.getlist('interests')
        if updated_interests:
            interests_vector_scibert = compute_interests_vector_scibert(updated_interests)
            interests_vector_fasttext = compute_interests_vector_fasttext(updated_interests)
            updated_data = {
                "name": request.form['name'],
                "surname": request.form['surname'],
                "age": request.form['age'],
                "gender": request.form['gender'],
                "interests": updated_interests,
                "interests_vector_scibert": interests_vector_scibert,
                "interests_vector_fasttext": interests_vector_fasttext,
            }
            users_collection.update_one({"_id": ObjectId(user['_id'])}, {"$set": updated_data})
            return redirect(url_for('logout'))
        else:
            return "Lütfen en az bir ilgi alanı seçiniz.", 400
@app.route('/search', methods=['GET'])
def search():
    query = request.args.get('query')
    if query:
        search_results = articles_collection.find({"$text": {"$search": query}})
        results = [{"name": result["name"], "title": result["title"]} for result in search_results]
    else:
        results = []
    return render_template('article.html', results=results)

def load_articles(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        articles = []
        for line in file:
            try:
                article = json.loads(line)
                articles.append(article)
            except json.JSONDecodeError:
                continue

        for article in articles:
            articles_collection.insert_one({
                "name": article["name"],
                "title": article["title"],
                "abstract": article["abstract"],
                "keywords": article["keywords"]
            })
@app.route('/submit-feedback', methods=['POST'])
def submit_feedback():

    user_email = session.get('user_email')
    if not user_email:
        return "Oturum açmanız gerekiyor", 403

    feedback_results = {}
    useful_articles = []
    not_useful_articles = []
    test=session.get("sbertcounter")
    test2=session.get("truecountsb")
    session["sbertcounter"] = test+5
    ece=0
    for key, value in request.form.items():
        if value == '1':  # Checkbox was checked
            article_name = key  # 'key' now directly represents the article's name
            feedback_results[article_name] = 'useful'
            useful_articles.append(article_name)
            ece+=1
        else:
            feedback_results[key] = 'not useful'


            # Update the user document with the list of useful articles
            user = users_collection.find_one({'email': user_email})
            if not user:
                return "Kullanıcı bulunamadı", 404

            for article_name in useful_articles:
                article = articles_collection.find_one({'name': article_name})
                if article:
                    article_title = article['title']
                    users_collection.update_one(
                        {'email': user_email},
                        {'$push': {'viewed_articles': article_title}}
                    )

    # Update the user document with the list of useful articles
    users_collection.update_one(
        {'email': user_email},
        {'$push': {'useful_articles': {'$each': useful_articles}}}
    )
    session["truecountsb"] = test2 + ece
    flash(f'Total count is now: {(test2 + ece)/(test+5)}', 'info')  # Flash mesajını kaydedin

    return redirect(url_for('article'))

  # or redirect to another page
def create_text_index():
    articles_collection.create_index([
        ("title", "text"),
        ("abstract", "text"),
        ("keywords", "text")
    ])

@app.route('/recommendations')
def recommendations():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    user = users_collection.find_one({"_id": ObjectId(session['user_id'])})
    if not user:
        return "Kullanıcı bulunamadı", 404

    if 'interests_vector_scibert' not in user:
        return "İlgi alanları bulunamadı veya kullanıcı mevcut değil", 404

    # Kullanıcının ilgi alanları vektörü ve görüntülenen makale başlıkları
    interest_vector = torch.tensor(user['interests_vector_scibert'])
    viewed_titles = user.get('viewed_articles', [])

    # Görüntülenen makale vektörlerini alın
    viewed_vectors = []
    for title in viewed_titles:
        article = articles_collection.find_one({'title': title})
        if article:
            viewed_vector = torch.tensor(get_sentence_vector(preprocess_text(article['abstract'])))
            viewed_vectors.append(viewed_vector)

    # Vektörleri birleştir ve ortalamasını al
    if viewed_vectors:
        all_vectors = torch.stack([interest_vector] + viewed_vectors)
        updated_interest_vector = torch.mean(all_vectors, dim=0)
    else:
        updated_interest_vector = interest_vector

    recommended_articles = recommend_articles_from_processed(updated_interest_vector, user.get("useful_articles", []))
    return render_template('recommendations.html', articles=recommended_articles)


@app.route('/recommendations_f')
def recommendations_f():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    user = users_collection.find_one({"_id": ObjectId(session['user_id'])})
    if not user:
        return "Kullanıcı bulunamadı", 404

    if 'interests_vector_fasttext' not in user:
        return "İlgi alanları bulunamadı veya kullanıcı mevcut değil", 404

    # Kullanıcının ilgi alanları vektörü ve görüntülenen makale başlıkları
    interest_vector = np.array(user['interests_vector_fasttext'])
    viewed_titles = user.get('viewed_articles', [])

    # Görüntülenen makale vektörlerini alın
    viewed_vectors = []
    for title in viewed_titles:
        article = articles_collection.find_one({'title': title})
        if article:
            viewed_vector = np.array(model.get_sentence_vector(preprocess_text(article['abstract'])))
            viewed_vectors.append(viewed_vector)

    # Vektörleri birleştir ve ortalamasını al
    if viewed_vectors:
        all_vectors = np.vstack([interest_vector] + viewed_vectors)
        updated_interest_vector = np.mean(all_vectors, axis=0)
    else:
        updated_interest_vector = interest_vector

    listread = user.get("useful_articles", [])
    recommended_articles = recommend_articles_ft(updated_interest_vector, listread)
    return render_template('recommendations_f.html', articles=recommended_articles)


@app.route('/article')
def article():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('article.html', results=[])

@app.route('/detail/<title>')
def article_detail(title):
    article = articles_collection.find_one({'title': title})
    if not article:
        return "Makale bulunamadı", 404
        # Kullanıcı giriş yapmış ise görüntülenen makaleyi kaydet
    if 'user_id' in session:
        user_id = session['user_id']
        users_collection.update_one(
            {'_id': ObjectId(user_id)},
            {'$push': {'viewed_articles': title}}
        )

    return render_template('detail.html', article=article)

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    return redirect(url_for('home'))

if __name__ == '__main__':

    create_text_index()
    app.run(port=5000)