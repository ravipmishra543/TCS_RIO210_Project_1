import matplotlib.pyplot as plt
import seaborn as sns

def plot_label_distribution(data, save_path=None):
    plt.figure(figsize=(10, 6))
    sns.countplot(x='label', data=data, palette='viridis')
    plt.title('Label Distribution')
    plt.xlabel('Labels')
    plt.ylabel('Count')
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_text_length_distribution(data, save_path=None):
    data['text_length'] = data['text'].apply(len)
    plt.figure(figsize=(10, 6))
    sns.histplot(data['text_length'], bins=30, kde=True, color='purple')
    plt.title('Text Length Distribution')
    plt.xlabel('Text Length')
    plt.ylabel('Frequency')
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_word_cloud(data, save_path=None):
    from wordcloud import WordCloud
    text = ' '.join(data['text'].tolist())
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud of Text Data')
    if save_path:
        plt.savefig(save_path)
    plt.show()
