# Bertemu dengan JANSEN: teman Anda

# impor pustaka yang diperlukan
import io
import random
import string # untuk memproses string python standar
import warnings
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('popular', quiet=True) # untuk mengunduh paket

# hapus komentar berikut hanya pada penggunaan pertama
#nltk.download('punkt') # hanya digunakan saat pertama kali
#nltk.download('wordnet') # hanya digunakan saat pertama kali

# Membaca korpus
with open('chatbot.txt','r', encoding='utf8', errors ='ignore') as fin:
    raw = fin.read().lower()

# Tokenisasi
sent_tokens = nltk.sent_tokenize(raw)# mengonversi ke daftar kalimat
word_tokens = nltk.word_tokenize(raw)# mengonversi ke daftar kata

# Pra-pemrosesan
lemmer = WordNetLemmatizer()
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))


# Pencocokan Kata Kunci
GREETING_INPUTS = ("halo", "hai", "assalamualaikum", "apa kabar", "hey", "baik")
GREETING_RESPONSES = {
    "halo": "Halo!",
    "hai": "Hai!",
    "assalamualaikum": "Waalaikumsalam!",
    "apa kabar": "Baik, Bagaimana denganmu?",
    "baik": "Bagus!",
    "hey": "Hey!"
}

def greeting(sentence):
    """Jika masukan pengguna adalah sapaan, kembalikan respons sapaan yang sesuai"""
    words = sentence.split()
    for word in words:
        if word.lower() in GREETING_INPUTS:
            return GREETING_RESPONSES[word.lower()]
    return None


# Daftar stop words bahasa Indonesia
indonesian_stop_words = ['dan', 'yang', 'untuk', 'pada', 'ke', 'karena', 'oleh', '?',...]

# Informasi tentang JANSEN dan chatbots
info_responses = {
    "siapa jansen": "JANSEN adalah chatbot berbasis NLTK yang dirancang untuk membantu dan berinteraksi dengan Anda dalam bahasa Indonesia.",
    "apa itu chatbot": "Chatbot adalah program komputer yang dirancang untuk mensimulasikan percakapan dengan pengguna manusia, terutama melalui internet.",
    "bagaimana jansen bekerja": "JANSEN bekerja dengan memproses bahasa alami yang Anda gunakan dan mencoba untuk memberikan respons yang paling relevan berdasarkan data yang telah diprogram sebelumnya."
}

def handle_info_question(user_response):
    """Menangani pertanyaan khusus tentang JANSEN atau chatbots secara umum"""
    user_response = user_response.lower()
    if user_response in info_responses:
        return info_responses[user_response]
    return None

# Data tebak-tebakan yang lebih lucu dan kreatif
tebak_tebakan = {
    "Apa yang bisa berjalan dan melompat tapi tidak pernah bergerak?": "jalan",
    "Apa yang lebih besar dari gajah tapi tidak berat sama sekali?": "bayangan gajah",
    "Apa yang dimiliki oleh semua orang, Anda bisa melihatnya, tetapi tidak bisa mereka?": "nama belakang",
    "Apa yang naik dan turun, tapi selalu di tempat yang sama?": "tangga",
    "Apa yang memiliki kunci tapi tidak bisa membuka pintu?": "piano",
    "Apa yang bisa dipegang tanpa menggunakan tangan?": "nafas",
    "Apa yang memiliki leher tapi tidak punya kepala?": "botol",
    "Apa yang bisa Anda tangkap tapi tidak bisa Anda lempar?": "pilek",
    "Apa yang memiliki banyak gigi tapi tidak bisa menggigit?": "sisir",
    "Apa yang bisa Anda buat, Anda bagikan, tetapi Anda tidak pernah bisa menyimpannya?": "janji"
}

def mulai_tebak_tebakan():
    pertanyaan = random.choice(list(tebak_tebakan.keys()))
    jawaban = tebak_tebakan[pertanyaan]
    return pertanyaan, jawaban

def handle_tebak_tebakan_answer(user_response, correct_answer):
    if user_response.lower() == correct_answer:
        return "Benar sekali! Anda menjawab dengan tepat."
    else:
        return "Salah! Jawabannya adalah " + correct_answer

# Kata kunci untuk tawa
LAUGH_INPUTS = ("haha", "hihi", "hehe", "lucu", "wkwk", "hoho")

def handle_laughter(sentence):
    """Jika masukan pengguna adalah tawa, kembalikan respons tawa yang sesuai"""
    words = sentence.split()
    for word in words:
        if word.lower() in LAUGH_INPUTS:
            return "Haha, senang melihat Anda tertawa!"
    return None

# Menghasilkan respons
def response(user_response):
    jansen_response = ''
    info_answer = handle_info_question(user_response)
    if info_answer:
        return info_answer
    if user_response in ['ayo main', 'main tebak-tebakan']:
        pertanyaan, jawaban = mulai_tebak_tebakan()
        jansen_response += "Oke, coba jawab tebak-tebakan ini: " + pertanyaan
        return jansen_response
    sent_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words=indonesian_stop_words)
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if req_tfidf == 0:
        jansen_response = jansen_response + "Maaf! Saya tidak mengerti yang anda katakan"
        return jansen_response
    else:
        jansen_response = jansen_response + sent_tokens[idx]
        return jansen_response


flag=True
print("JANSEN: Nama saya JANSEN. Saya adalah chatbot yang akan menemani anda. Jika Anda ingin keluar, ketik keluar!")
while(flag==True):
    user_response = input()
    user_response=user_response.lower()
    if(user_response!='keluar'):
        if(user_response=='terimakasih' or user_response=='terima kasih' ):
            flag=False
            print("JANSEN: Terima kasih Kembali..")
        elif user_response == 'main tebak-tebakan':
            pertanyaan, jawaban = mulai_tebak_tebakan()
            print("JANSEN: " + pertanyaan)
            user_response = input().lower()
            print("JANSEN: " + handle_tebak_tebakan_answer(user_response, jawaban))
        else:
            if(greeting(user_response)!=None):
                print("JANSEN: "+greeting(user_response))
            elif handle_laughter(user_response) != None:
                print("JANSEN: "+handle_laughter(user_response))
            else:
                print("JANSEN: ",end="")
                print(response(user_response))
                if user_response in sent_tokens:
                    sent_tokens.remove(user_response)
    else:
        flag=False
        print("JANSEN: Selamat tinggal! Sampai Bertemu Lagi..")    
