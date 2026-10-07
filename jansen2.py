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
GREETING_INPUTS = ("halo", "hai", "assalamualaikum", "apa kabar", "baik", "udah ngopi belum", "udah dong", "hey")
GREETING_RESPONSES = {
    "halo": "Halo!",
    "hai": "Hai!",
    "assalamualaikum": "Waalaikumsalam!",
    "apa kabar": "Baik, Bagaimana denganmu?",
    "baik": "Bagus!",
    "udah ngopi belum": "Udah dong, kamu udah belum ?",
    "udah dong": "Mantapp!",
    "hey": "Hey!"
}

def greeting(sentence):
    """Jika masukan pengguna adalah sapaan, kembalikan respons sapaan yang sesuai"""
    for greeting in GREETING_INPUTS:
        if greeting.lower() in sentence.lower():
            return GREETING_RESPONSES[greeting.lower()]
    return None


# Daftar stop words bahasa Indonesia yang lebih lengkap (120 kata utama)
indonesian_stop_words = [
    'ada', 'adalah', 'adanya', 'adapun', 'agak', 'agaknya', 'agar', 'akan', 'akankah', 'akhir',
    'akhiri', 'akhirnya', 'aku', 'akulah', 'amat', 'amatlah', 'anda', 'andalah', 'antar', 'antara',
    'antaranya', 'apa', 'apaan', 'apabila', 'apakah', 'apalagi', 'apatah', 'artinya', 'asal', 'asalkan',
    'atas', 'atau', 'ataukah', 'ataupun', 'awal', 'awalnya', 'bagai', 'bagaikan', 'bagaimana', 'bagaimanakah',
    'bagaimanapun', 'bagi', 'bagian', 'bahkan', 'bahwa', 'bahwasanya', 'baik', 'bakal', 'bakalan', 'balik',
    'banyak', 'bapak', 'baru', 'bawah', 'beberapa', 'begini', 'beginian', 'beginikah', 'beginilah', 'begitu',
    'begitukah', 'begitulah', 'begitupun', 'bekerja', 'belakang', 'belakangan', 'belum', 'belumlah', 'benar',
    'benarkah', 'benarlah', 'berada', 'berakhir', 'berakhirlah', 'berakhirnya', 'berapa', 'berapakah', 'berapalah',
    'berapapun', 'berarti', 'berawal', 'berbagai', 'berdatangan', 'beri', 'berikan', 'berikut', 'berikutnya', 'berjumlah',
    'berkali-kali', 'berkata', 'berkehendak', 'berkeinginan', 'berkenaan', 'berlainan', 'berlalu', 'berlangsung', 'berlebihan',
    'bermacam', 'bermacam-macam', 'bermaksud', 'bermula', 'bersama', 'bersama-sama', 'bersiap', 'bersiap-siap', 'bertanya',
    'bertanya-tanya', 'berturut', 'berturut-turut', 'bertutur', 'berujar', 'berupa', 'besar', 'betul', 'betulkah', 'biasa',
    'biasanya', 'bila', 'bilakah', 'bisa', 'bisakah', 'boleh', 'bolehkah', 'bolehlah', 'buat', 'bukan', 'bukankah', 'bukanlah', 'bukannya'
]
# Informasi tentang JANSEN dan chatbots
info_responses = {
    "siapa jansen": "JANSEN adalah chatbot berbasis NLTK yang dirancang untuk membantu dan berinteraksi dengan Anda dalam bahasa Indonesia.",
    "apa itu chatbot": "Chatbot adalah program komputer yang dirancang untuk mensimulasikan percakapan dengan pengguna manusia, terutama melalui internet.",
    "bagaimana jansen bekerja": "JANSEN bekerja dengan memproses bahasa alami yang Anda gunakan dan mencoba untuk memberikan respons yang paling relevan berdasarkan data yang telah diprogram sebelumnya.",
    "bagaimana jansen bisa berbahasa indonesia": "JANSEN bisa berbahasa indonesia karena menggunakan bahasa alami yang Anda gunakan dan mencoba untuk memberikan respons yang paling relevan berdasarkan data yang telah diprogram sebelumnya.",
    "apakah kamu kenal bu esi": "Bu Esi adalah seorang dosen yang baik dan sabar, dia adalah seorang dosen di UNISA YOGYAKARTA.",
    "apakah kamu kenal mas ariva": "Sepertinya tidak asing di telinga saya, dia adalah seorang mas-mas biasa saja.",
    "apakah kamu kenal pablo": "Pablo adalah seorang pemuda yang baik hati dan tampan rupawan, dia adalah seorang mahasiswa di UNISA YOGYAKARTA.",
}

def handle_info_question(user_response):
    """Menangani pertanyaan khusus tentang JANSEN atau chatbots secara umum"""
    user_response = user_response.lower()
    if user_response in info_responses:
        return info_responses[user_response]
    return None

# Data tebak-tebakan yang lebih banyak, beragam, dan menggunakan bahasa non-formal
tebak_tebakan = {
    "Apa yang bisa jalan-jalan tapi nggak pernah gerak sedikit pun?": "jalan raya",
    "Apaan tuh yang lebih gede dari gajah tapi nggak berat sama sekali?": "bayangannya gajah",
    "Apa yang lo punya, gue bisa liat, tapi lo sendiri nggak bisa liat?": "nama belakang",
    "Apaan tuh yang naik turun mulu, tapi tetep aja di tempat yang sama?": "lift",
    "Apa yang punya kunci banyak tapi nggak bisa buka pintu?": "piano",
    "Apa yang bisa lo pegang tanpa pake tangan?": "napas",
    "Apa yang punya leher tapi nggak punya kepala?": "botol",
    "Apa yang bisa lo tangkep tapi nggak bisa lo lempar?": "flu",
    "Apa yang giginya banyak banget tapi nggak bisa gigit?": "sisir",
    "Apa yang bisa lo bikin, lo kasih ke orang, tapi nggak bisa lo simpen?": "janji",
    "Apaan tuh yang jatoh mulu tapi nggak pernah sakit?": "hujan",
    "Apa yang punya mata tapi buta total?": "jarum",
    "Apa yang bisa lari kenceng tapi nggak pernah jalan?": "air",
    "Apa yang punya kepala dan ekor tapi nggak punya badan?": "koin",
    "Apa yang bisa terbang tanpa sayap?": "gosip",
    "Kenapa ayam nyebrang jalan?": "pengen ke seberang",
    "Apa yang dipake di kaki tapi nggak bisa jalan?": "kaus kaki",
    "Kenapa pohon kelapa tinggi banget?": "pengen liat pemandangan",
    "Apa yang bisa bikin orang nangis tanpa nyakitin mereka?": "bawang",
    "Kenapa ikan hidup di air?": "karena nggak bisa napas di darat"
}

def mulai_tebak_tebakan():
    pertanyaan, jawaban = random.choice(list(tebak_tebakan.items()))
    return pertanyaan, jawaban

def handle_tebak_tebakan_answer(user_response, correct_answer):
    # Hapus stopwords dari jawaban pengguna
    user_words = user_response.lower().split()
    filtered_user_response = ' '.join([word for word in user_words if word not in indonesian_stop_words])
    
    # Bandingkan dengan jawaban yang benar
    if filtered_user_response == correct_answer.lower():
        return "Benar sekali! Anda menjawab dengan tepat.", True
    elif user_response.lower() in ["menyerah", "nyerah"]:
        return f"Jawabannya adalah {correct_answer}.", True
    else:
        return "Maaf, jawaban Anda kurang tepat. Coba lagi atau katakan 'Menyerah' jika ingin tahu jawabannya.", False

# Kata kunci untuk tawa yang lebih banyak (25 kata tertawa yang biasa digunakan)
LAUGH_INPUTS = (
    "haha", "wkwk", "hihi", "hehe", "wahaha", "wihihi", "wawawa", "wkwkwk", 
    "hahaha", "hihihi", "hehehe", "hohoho", "huhuhu", "wakaka", "wkakaka",
    "hahahaha", "wkwkwkwk", "hehehehe", "huehue", "kekeke", "xixixi",
    "bwahaha", "ahaha", "ihihi", "ehehe"
)

def handle_laughter(sentence):
    """Jika masukan pengguna adalah tawa, kembalikan respons tawa yang sesuai"""
    words = sentence.split()
    for word in words:
        if word.lower() in LAUGH_INPUTS:
            return "Haha, senang melihat Anda tertawa!"
    return None

# Daftar fakta acak yang diperbanyak
RANDOM_FACTS = [
    "Chatbot pertama bernama ELIZA diciptakan pada tahun 1966.",
    "Beberapa chatbot modern menggunakan kecerdasan buatan dan pembelajaran mesin.",
    "Chatbot dapat digunakan di berbagai industri, termasuk layanan pelanggan dan kesehatan.",
    "Saya adalah chatbot berbasis NLTK yang dirancang untuk berbahasa Indonesia.",
    "Chatbot terus berkembang dan menjadi semakin canggih seiring waktu.",
    "Indonesia punya lebih dari 17.000 pulau, tapi cuma sekitar 6.000 yang berpenghuni.",
    "Borobudur itu candi Buddha terbesar di dunia, dibangun pada abad ke-9.",
    "Komodo cuma ada di Indonesia, tepatnya di pulau Komodo, Rinca, Flores, dan Gili Motang.",
    "Bahasa Indonesia punya lebih dari 700 bahasa daerah yang masih aktif dipakai.",
    "Rafflesia arnoldii, bunga terbesar di dunia, bisa ditemuin di hutan Sumatera dan Kalimantan.",
    "Wayang kulit udah diakui UNESCO sebagai Warisan Budaya Dunia sejak 2003.",
    "Indonesia punya 150 gunung berapi aktif, paling banyak di dunia.",
    "Orangutan cuma bisa ditemuin di hutan Sumatera dan Kalimantan.",
    "Batik Indonesia udah diakui UNESCO sebagai Warisan Budaya Dunia sejak 2009.",
    "Taman Nasional Bunaken di Sulawesi Utara punya 70% dari semua spesies ikan di Indo-Pasifik.",
    "Candi Prambanan itu candi Hindu terbesar di Asia Tenggara, dibangun pada abad ke-9.",
    "Indonesia punya lebih dari 300 kelompok etnis yang berbeda-beda.",
    "Rendang, makanan khas Sumatera Barat, pernah dinobatkan sebagai makanan terenak di dunia.",
    "Tari Saman dari Aceh udah diakui UNESCO sebagai Warisan Budaya Dunia.",
    "Indonesia adalah negara kepulauan terbesar di dunia.",
    "Bahasa Indonesia berasal dari bahasa Melayu yang udah dipakai sebagai lingua franca di Nusantara selama berabad-abad.",
    "Garis Wallace, batas biogeografis yang memisahkan fauna Asia dan Australia, melintasi Indonesia.",
    "Danau Toba di Sumatera Utara adalah danau vulkanik terbesar di dunia.",
    "Gamelan, musik tradisional Jawa dan Bali, udah mempengaruhi banyak komposer musik klasik Barat.",
    "Indonesia punya lebih dari 400 bandara, termasuk bandara terapung pertama di dunia di Bali."
]

def random_fact():
    return random.choice(RANDOM_FACTS)

# Daftar tips acak yang diperbanyak
TIPS = [
    "Jangan lupa minum air yang cukup hari ini!",
    "Istirahat sejenak dari layar komputer bisa membantu mengurangi lelah mata.",
    "Olahraga ringan setiap hari bisa meningkatkan mood dan produktivitas.",
    "Jangan ragu untuk bertanya jika ada yang tidak Anda pahami.",
    "Belajar hal baru setiap hari bisa membantu mengembangkan diri.",
    "Jangan lupa minum air yang cukup hari ini!",
    "Istirahat sejenak dari layar komputer bisa membantu mengurangi lelah mata.",
    "Olahraga ringan setiap hari bisa meningkatkan mood dan produktivitas.",
    "Jangan ragu untuk bertanya jika ada yang tidak Anda pahami.",
    "Belajar hal baru setiap hari bisa membantu mengembangkan diri.",
    "Sarapan sehat bisa meningkatkan konsentrasi dan energi sepanjang hari.",
    "Tidur cukup (7-9 jam) penting untuk kesehatan fisik dan mental.",
    "Meditasi selama 10 menit sehari bisa mengurangi stres dan kecemasan.",
    "Baca buku minimal 30 menit sehari untuk meningkatkan pengetahuan.",
    "Terapkan '3R': Reduce, Reuse, Recycle untuk menjaga lingkungan.",
    "Gunakan transportasi umum atau sepeda untuk mengurangi polusi.",
    "Makan lebih banyak sayur dan buah untuk meningkatkan sistem kekebalan tubuh.",
    "Lakukan peregangan setiap 1-2 jam saat bekerja di depan komputer.",
    "Simpan sebagian penghasilan untuk dana darurat dan masa depan.",
    "Batasi penggunaan media sosial untuk mengurangi stres dan kecemasan.",
    "Praktikkan rasa syukur setiap hari untuk meningkatkan kebahagiaan.",
    "Jaga kebersihan tangan dengan sering mencuci tangan atau menggunakan hand sanitizer.",
    "Gunakan sunscreen saat beraktivitas di luar ruangan untuk melindungi kulit.",
    "Konsumsi makanan lokal untuk mendukung petani dan ekonomi setempat.",
    "Belajar bahasa daerah bisa membantu memahami budaya lokal lebih baik."
]

def random_tip():
    return random.choice(TIPS)

# Daftar kosakata untuk memulai random fact dan random tip
FACT_KEYWORDS = ["berikan fakta", "fakta random", "fakta acak", "fakta menarik", "tahukah kamu"]
TIP_KEYWORDS = ["berikan tips", "tips random", "tips acak", "saran", "nasihat"]

# Menghasilkan respons
def response(user_response):
    jansen_response = ''
    info_answer = handle_info_question(user_response)
    if info_answer:
        return info_answer
    if user_response in ['ayo main', 'main tebak-tebakan']:
        print("JANSEN: Ayo main tebak-tebakan! Katakan 'Menyerah' atau 'Nyerah' jika ingin tahu jawabannya.")
        while True:
            pertanyaan, jawaban = mulai_tebak_tebakan()
            print("JANSEN: " + pertanyaan)
            selesai = False
            while not selesai:
                user_response = input(f"{user_name}: ").lower()
                response, selesai = handle_tebak_tebakan_answer(user_response, jawaban)
                print("JANSEN: " + response)
            user_response = input(f"{user_name}: Mau main lagi? (ya/tidak) ").lower()
            if user_response != 'ya':
                break
        return ""
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
        return jansen_response.replace("Anda", user_name)


user_name = ""

# Di awal percakapan
print("JANSEN: Nama saya JANSEN. Boleh saya tahu nama Anda?")
user_name = input(f"{user_name}: ").strip()
print(f"JANSEN: Senang berkenalan dengan Anda, {user_name}! Saya adalah chatbot yang akan menemani Anda. Jika Anda ingin keluar, ketik keluar!")

message_count = 0

flag=True
while(flag==True):
    user_response = input(f"{user_name}: ").lower()
    if(user_response!='keluar'):
        if(user_response=='terimakasih' or user_response=='terima kasih' ):
            flag=False
            print(f"JANSEN: Terima kasih kembali, {user_name}.")
        elif user_response == 'main tebak-tebakan':
            print("JANSEN: Ayo main tebak-tebakan! Katakan 'Menyerah' atau 'Nyerah' jika ingin tahu jawabannya.")
            while True:
                pertanyaan, jawaban = mulai_tebak_tebakan()
                print("JANSEN: " + pertanyaan)
                selesai = False
                while not selesai:
                    user_response = input(f"{user_name}: ").lower()
                    response, selesai = handle_tebak_tebakan_answer(user_response, jawaban)
                    print("JANSEN: " + response)
                user_response = input(f"{user_name}: Mau main lagi? (ya/tidak) ").lower()
                if user_response != 'ya':
                    break
        elif any(keyword in user_response for keyword in FACT_KEYWORDS):
            print("JANSEN: " + random_fact())
        elif any(keyword in user_response for keyword in TIP_KEYWORDS):
            print("JANSEN: " + random_tip())
        elif user_response == 'fakta':
            print("JANSEN: " + random_fact())
        elif user_response == 'tips':
            print("JANSEN: " + random_tip())
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
        print(f"JANSEN: Selamat tinggal, {user_name}! Sampai bertemu lagi.")    