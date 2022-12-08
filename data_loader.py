import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
import time
#import pattern
import warnings
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
from tqdm import tqdm_notebook as tqdm
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')

#import tqdm.tqdm.notebook as tqdm
from ignore_words import *
try:
    from gensim.utils import lemmatize
except Exception as e:
    from gensim.utils import lemmatize
from gensim.utils import lemmatize
xml_path = 'ABSA-15_Restaurants_Train_Final.xml'
print(lemmatize)
def parse_data_2015(xml_path):
    container = []                                              
    reviews = ET.parse(xml_path).getroot()                      
    for review in reviews:  
        sentences = list(list(review)[0])       
        for sentence in sentences:                                  
            sentence_text = list(sentence)[0].text          
            try:                                                     
                opinions = list(list(sentence)[1])
                for opinion in opinions:                                
                    polarity = opinion.attrib["polarity"]
                    target = opinion.attrib["target"]
                    row = {"sentence": sentence_text, "sentiment":polarity}   
                    container.append(row)                                                              
            except IndexError: 
                row = {"sentence": sentence_text}        
                container.append(row)                                                               
    return pd.DataFrame(container)

def cleaning_function(tips):
    all_ = []
    for tip in tqdm(tips):
        time.sleep(0.0001)
#       Decoding function
#        decode = tip.decode("utf-8-sig")
#       Lowercasing before negation
        lower_case = tip.lower()
#       Replace apostrophes with words
        words = lower_case.split()
        split = [words_list[word] if word in words_list else word for word in words]
        reformed = " ".join(split)
        for i in range(4):
            try:
                lemm = lemmatize(lower_case)
            except Exception as e:
                continue
        lemm = [x.decode('UTF8') for x in lemm]
        all_.append(lemm)
        
    return all_      


def separate_word_tag(df_lem_test):
    words=[]
    types=[]
    df= pd.DataFrame()
    for row in df_lem_test:
        sent = []
        type_ =[]
        for word in row:
            split = word.split('/')
            sent.append(split[0])
            type_.append(split[1])
        words.append(' '.join(word for word in sent))
        types.append(' '.join(word for word in type_))
    df['lem_words']= words
    df['lem_tags']= types
    return df

ABSA_df = parse_data_2015(xml_path)
print(ABSA_df.head())
print ("Original:", ABSA_df.shape)
dd = ABSA_df.drop_duplicates()
ABSA_dd = dd.reset_index(drop=True)
print("Drop Dupicates:", ABSA_dd.shape)
print(ABSA_dd.sentiment.value_counts())
dd_dn = ABSA_dd.fillna('neutral')

df = dd_dn.reset_index(drop=True)
df.sentiment.value_counts()
word_tag = cleaning_function(df.sentence)
lemm_df = separate_word_tag(word_tag)
df_training = pd.concat([df, lemm_df], axis=1)
df_training['word_tags'] = word_tag
print(df_training.head())
df_training['sentiment'] = df_training.sentiment.map(lambda x: int(2) if x =='positive' else int(0) if x =='negative' else int(1) if x == 'neutral' else np.nan)
df_training = df_training.reset_index(drop=True)
print(df_training[df_training['lem_words']==''])
df_training = df_training.drop([475, 648, 720])
df_training = df_training.reset_index(drop=True)


train, test = train_test_split(df_training, test_size=0.3, random_state=1)

t_1 = train[train['sentiment']==1].sample(800,replace=True)
t_2 = train[train['sentiment']==2].sample(800,replace=True)
t_3 = train[train['sentiment']==0].sample(800,replace=True)
training_bs = pd.concat([t_1, t_2, t_3])

# sanity check 
df_training.shape[0] == (train.shape[0] + test.shape[0])

training_bs = training_bs.reset_index(drop=True)
training_bs.to_csv('training_bs.csv', header=True, index=False, encoding='UTF8')
test = test.reset_index(drop=True)
test.to_csv('testing.csv', header=True, index=False, encoding='UTF8')