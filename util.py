#正确。获得句向量
import numpy as np


def get_word_embeddings(word_embeddings_file):
    word_embeddings = {}
    f = open(word_embeddings_file, encoding='utf-8')
    for line in f:
        # 把第一行的内容去掉
        if '130 400\n' not in line:
            values = line.split()
            # 第一个元素是词语
            word = values[0]
            embedding = np.asarray(values[1:], dtype='float32')
            word_embeddings[word] = embedding
    f.close()
    print("一共有" + str(len(word_embeddings)) + "个词语/字。")
    return word_embeddings

def get_sentences(sentences_file):
    with open(sentences_file,'r') as fs:
         sentences=fs.readlines()
    return sentences

def get_sentence_vector(sentences,word_embeddings,stopwords=None):#sentences is list
        #if stopwords is None:
        #   stopwords = []
        #clean_sentences = [self.remove_stopwords(r,stopwords) for r in sentences]
        sentence_vectors = []    #句子向量
        #for i in clean_sentences:
        for i in sentences:
            if len(i) != 0:
                #v = sum([word_embeddings.get(w, np.zeros((1,))) for w in i.split()]) / (len(i.split()) + 0.001)
               # v = sum([word_embeddings.get(w, np.zeros((1,))) for w in i]) / (len(i) + 0.001)
                v = sum([word_embeddings.get(w, np.random.uniform(0, 1, 400)) for w in i]) / (len(i) + 0.001)
            else:
              #  v = np.zeros((30,))
                v = np.random.uniform(0, 1, 400)
            sentence_vectors.append(v)
        return sentence_vectors

def get_sent_vec(word_embeddings_file,sentences_file):
    word_embeddings=get_word_embeddings(word_embeddings_file)
    sentences=get_sentences(sentences_file)
    sent_vec=get_sentence_vector(sentences,word_embeddings)
    return sent_vec




params_word_embeddings_file='/home/ddr/data4/vec/params_word2vec2.vector'
params_sentences_file='/home/ddr/data4/trainData/params.txt'
params_sent_vec=get_sent_vec(word_embeddings_file,sentences_file)
methods_word_embeddings_file='/home/ddr/data4/vec/methods_word2vec2.vector'
methods_sentences_file='/home/ddr/data4/trainData/methods.txt'
methods_sent_vec=get_sent_vec(methods_word_embeddings_file,methods_sentences_file)
urls_word_embeddings_file='/home/ddr/data4/vec/urls_word2vec2.vector'
urls_sentences_file='/home/ddr/data4/trainData/urls.txt'
urls_sent_vec=get_sent_vec(urls_word_embeddings_file,urls_sentences_file)

seg_sent_vec=[]
for i in range(len(sent_vec)):
    a_sent_vec=(params_sent_vec[i]+methods_sent_vec[i]+urls_sent_vec)/3
    seg_sent_vec.append(a_sent_vec)

dates_word_embeddings_file='/home/ddr/data4/vec/dates_word2vec2.vector'
dates_sentences_file='/home/ddr/data4/trainData/dates.txt'
dates_sent_vec=get_sent_vec(dates_word_embeddings_file,dates_sentences_file)


all_sent_vec=[]
for i in range(len(sent_vec)):
    a_sent_vec=(seg_sent_vec[i]+dates_sent_vec[i])/2
    all_sent_vec.append(a_sent_vec)

a='ss'