import jieba
from jieba import posseg
from sklearn.feature_extraction.text import CountVectorizer
import lda
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

class SplitComment:
    def __init__(self, percent_list):
        self.percent_list=percent_list
        
    def __iter__(self):
        for percent in self.percent_list:
            yield cut_words_with_pos(percent)
            
            
def split_by_percent(filepath):
    text=open(filepath).read()
    percent_list=[]
    for line in text:
        line = text.split('\n\n')
        percent_list.append(line)
    return percent_list


def cut_words_with_pos(text):
    seg=jieba.posseg.cut(text)
    res=[]
    for i in seg:
        if i.flag in ["a", "v", "x", "n", "an", "vn", "nz", "nt", "nr"] and is_fine_word(i.word):
            res.append(i.word)
    return list(res)

def is_fine_word(word, min_length=2):
    Stop_words=set([w.strip() for w in open('PATH_TO_STOPWORDS').readlines()])
    if len(word)>=min_length and word not in Stop_words:
        return True
    else:
        return False

def seg_word(str):
    lines=open("./combined.txt",'r').readlines()

    wordlist=[]
    
    for line in lines:
        if str in line:
            words=cut_words_with_pos(line)
            wordlist+=words
    return(wordlist)

def get_lda_input(percents):
    corpus = [" ".join(word_list) for word_list in percents]
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(corpus)
    return(X.toarray(), vectorizer)
  
def plot_topic(doc_topic):
   f, ax = plt.subplots(figsize=(10, 4))
   cmap = sns.cubehelix_palette(start=1, rot=3, gamma=0.8, as_cmap=True)
   sns.heatmap(doc_topic, cmap=cmap, linewidths=0.05, ax=ax)
   ax.set_title('Proportion Per Topic in Every Part of the Movie')
   ax.set_xlabel('Topic')
   ax.set_ylabel('Percent')
   plt.show()
   f.savefig('./topic_heatmap.jpg', bbox_inches='tight') 
 
def lda_train(weight, vectorizer):
    model = lda.LDA(n_topics=20, n_iter=10, random_state=1)
    model.fit(weight)

    doc_num = len(weight)
    topic_word = model.topic_word_
    vocab = vectorizer.get_feature_names()
    titles = ["{}percent".format(i) for i in range(1, doc_num + 1)]

    n_top_words = 20
    for i, topic_dist in enumerate(topic_word):
        topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words + 1):-1]
        print('Topic {}: {}'.format(i, ' '.join(topic_words)))

    doc_topic = model.doc_topic_
    print(doc_topic, type(doc_topic))
    plot_topic(doc_topic)
    for i in range(doc_num):
        print("{} (top topic: {})".format(titles[i], np.argsort(doc_topic[i])[:-4:-1]))
        
 def main():
    percent_list = split_by_percent("./combined.txt")
    percents = SplitComment(percent_list)
    weight, vectorizer = get_lda_input(percents)
    lda_train(weight, vectorizer)
 
 if __name__ == "__main__":
    main()
