import fasttext.util
fasttext.util.download_model('en', if_exists='ignore')
# English
ft = fasttext.load_model('cc.en.300.bin')