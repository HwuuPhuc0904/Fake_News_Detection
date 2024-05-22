import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
sns.set_theme(style='darkgrid')

from sklearn.pipeline import Pipeline
# for tfidf 
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

def convert_label_to_text(n):
    if n == 1:
        return "UNRELIABLE"
    elif n == 0:
        return "RELIABLE"
    else:
        return "No label"

def count_plot(data, name='label'):
    sns.countplot(x=name, data=data)
    plt.show()

    labels= data[name].unique()
    num_labels = []
    for label in labels:
        num_labels.append((data[name] == label).sum())
    
    n = len(data)
    for (label, num_label) in zip(labels, num_labels):
        print(f'Percent of {label} label: {round(num_label/n *100, 2)}%')

def correlation_plot(data, cols=['label']):
    subnets = data[cols]
    subnets.corr(method='pearson').style.background_gradient(cmap='coolwarm', axis=None).set_precision(2)


def box_plot(data, name='label'):
    return 

def box_plot(data, name='label'):
    return

def text_tfidf(data):
    pipeline = Pipeline([('count',CountVectorizer()),('tfidf',TfidfTransformer())])

    data_tfidf = pipeline.fit_transform(data).toarray()

    return pipeline, data_tfidf

def get_stop_words(stopwords_file = "vietnamese-stopwords/vietnamese-stopwords.txt"):
    with open(stopwords_file, "r", encoding='utf8') as file_sw:
        stop_word = file_sw.read().split(sep='\n')
        # đưa về word dạng "xin chào" thành "xin_chào" tương ứng với bộ tokenize của vncorenlp
        stop_word = [word.replace(" ", "_") for word in stop_word]
    return stop_word