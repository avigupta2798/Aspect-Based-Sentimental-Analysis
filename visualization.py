from data_loader import  *

from wordcloud import WordCloud
import matplotlib.pyplot as plt
from os import path
from PIL import Image

plt.hist(df_training.sentiment, bins = 3, align= 'mid')
plt.xticks(range(3), ['Negative','Neutral', 'Positive'])
plt.xlabel('Sentiment of Reviews')
plt.title('Distribution of Sentiment')
plt.show()


train_s0 = training_bs[training_bs.sentiment ==0]
train_s1 = training_bs[training_bs.sentiment ==1]
train_s2 = training_bs[training_bs.sentiment ==2]

all_text = ' '.join(word for word in train_s0.lem_words)

# Polarity == 0 negative
wordcloud = WordCloud(colormap='Reds', width=1000, height=1000, mode='RGBA', background_color='white').generate(all_text)
plt.figure(figsize=(20,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.margins(x=0, y=0)
plt.show()

# Polarity == 1 neutral
all_text = ' '.join(word for word in train_s1.lem_words)
wordcloud = WordCloud(width=1000, height=1000, colormap='Wistia', background_color='white', mode='RGBA').generate(all_text)
plt.figure( figsize=(20,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.margins(x=0, y=0)
plt.show()

# Polarity == 2 positive
all_text = ' '.join(word for word in train_s2.lem_words)
wordcloud_p2 = WordCloud(width=1000, height=1000, colormap='summer',background_color='white', mode='RGBA').generate(all_text)
plt.figure(figsize=(20,10))
plt.imshow(wordcloud_p2, interpolation='bilinear')
plt.axis("off")
plt.margins(x=0, y=0)
plt.show()

