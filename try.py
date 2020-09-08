#正确。使用 词向量文件 和 文本文件 获得 文本聚类结果
import numpy as np
from sklearn.cluster import KMeans
import numpy as np
#例子data = np.random.rand(100, 8) #生成一个随机数据，样本大小为100, 特征数为8
def get_word_embeddings(no_str,word_embeddings_file):#获得word_embeddings
    word_embeddings = {}
    f = open(word_embeddings_file, encoding='utf-8')
    for line in f:
        # 把第一行的内容去掉
        if no_str not in line:
            values = line.split()
            # 第一个元素是词语
            word = values[0]
            embedding = np.asarray(values[1:], dtype='float32')
            word_embeddings[word] = embedding
    f.close()
    print("一共有" + str(len(word_embeddings)) + "个词语/字。")
    return word_embeddings

def get_sentences(sentences_file):#获得句子 list
    with open(sentences_file,'r') as fs:
         sentences=fs.readlines()
    return sentences

def get_sentence_vector(sentences,word_embeddings,stopwords=None):#sentences is list ，获得句向量
        sentence_vectors = []    #句子向量
        for i in sentences:
            if len(i) != 0:
                v = sum([word_embeddings.get(w, np.random.uniform(0, 1, 400)) for w in i]) / (len(i) + 0.001)
            else:
                v = np.random.uniform(0, 1, 400)
            sentence_vectors.append(v)
        return sentence_vectors

def get_sent_vec(no_str,word_embeddings_file,sentences_file):    
    word_embeddings=get_word_embeddings(no_str,word_embeddings_file)
    sentences=get_sentences(sentences_file)
    sent_vec=get_sentence_vector(sentences,word_embeddings)
    return sent_vec


#------------------------------------文件路径参数-----------------------------------------------------------
#.vector和.txt文件路径
params_word_embeddings_file='/home/ddr/data4/vec/params_word2vec2.vector'#params 的vector 的文件路径
params_sentences_file='/home/ddr/data4/trainData/params.txt'#params 的文本的 文件路径
methods_word_embeddings_file='/home/ddr/data4/vec/methods_word2vec2.vector'#methods 的vector 的文件路径
methods_sentences_file='/home/ddr/data4/trainData/methods.txt'#methods 的文本的 文件路径
urls_word_embeddings_file='/home/ddr/data4/vec/url_word2vec2.vector'#url 的vector 的文件路径
urls_sentences_file='/home/ddr/data4/trainData/urls.txt'#url 的文本的 文件路径
dates_word_embeddings_file='/home/ddr/data4/vec/dates_word2vec2.vector'#dates 的vector 的文件路径
dates_sentences_file='/home/ddr/data4/trainData/dates.txt'#dates 的文本的 文件路径
result_file='/home/ddr/data4/label.txt'#最终聚类结果输出的文件路径


#----------------------------------获得句向量------------------------------------------------------
#获得params句向量
params_str='130 400'
params_sent_vec=get_sent_vec(params_str,params_word_embeddings_file,params_sentences_file)
#获得methods句向量
methods_str='4 400'
methods_sent_vec=get_sent_vec(methods_str,methods_word_embeddings_file,methods_sentences_file)
#获得url句向量
url_str='769 400'
urls_sent_vec=get_sent_vec(url_str,urls_word_embeddings_file,urls_sentences_file)

#获得url、methods、params共同作为操作行为的句向量
seg_sent_vec=[]#操作参数为：url、methods和params的三个加权平均
for i in range(len(params_sent_vec)):
    a_sent_vec=(params_sent_vec[i]+methods_sent_vec[i]+urls_sent_vec)/3
    seg_sent_vec.append(a_sent_vec)

#获得dates句向量
dates_str='23915 400'
dates_sent_vec=get_sent_vec(dates_str,dates_word_embeddings_file,dates_sentences_file)

#获得dates、操作行为 共同作为一条消息的句向量
all_sent_vec=[]#共为：dates和操作参数的加权平均
for i in range(len(seg_sent_vec)):
    a_sent_vec=(seg_sent_vec[i]+dates_sent_vec[i])/2
    all_sent_vec.append(a_sent_vec)

#-------------------------------------------聚类--------------------------------------------------------------
data=all_sent_vec#数据
#构造一个聚类数为100的聚类器
estimator = KMeans(n_clusters=100)#构造聚类器，分为100类
estimator.fit(data)#聚类
label_pred = estimator.labels_ #获取聚类标签
centroids = estimator.cluster_centers_ #获取聚类中心
inertia = estimator.inertia_ # 获取聚类准则的总和

with open(result_file,'w') as fw:#聚类结果写入
    for i,label in enumerate(label_pred):
        fw.write(str(i)+' '+str(label))

