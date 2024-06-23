import tkinter as tk
from tkinter import simpledialog, filedialog
import re
import sys
import pandas as pd
import pymorphy2
from nltk import word_tokenize, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
import torch
from torch import nn
import torchvision.models as models
from torch.utils.data import DataLoader, TensorDataset, RandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
import os
import nltk
from tkinter import scrolledtext
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('corpora')
nltk.download('wordnet')
from bert_dataset import CustomDataset
from bert_classifier import BertClassifier


def write_to_text_area(text):
    text_area.insert('end', text + '\n')


def collectFilesToDir(dir: str):
    fdDict1 = {
        'index': [],
        'text': [],
        'class': [],
        'category': [],
    }
    fdDict2 = {
        'index': [],
        'text': [],
        'class': [],
        'category': [],
    }
    category_dict1 = {}
    category_dict2 = {}
    category_count1 = 0
    category_count2 = 0

    for root, dirs, files in os.walk(dir):
        if 'test' in root:
            for file in files:
                class_name = os.path.basename(root)
                if class_name not in category_dict1:
                    category_dict1[class_name] = category_count1
                    category_count1 += 1
                category = category_dict1[class_name]
                print(f'{file}')
                with open(f'{root}/{file}', 'r', encoding='utf-8') as inFile:
                    text = inFile.read()
                fdDict1['index'].append(len(fdDict1['index']))
                fdDict1['text'].append(text)
                fdDict1['class'].append(class_name)
                fdDict1['category'].append(category)
        elif 'train' in root:
            for file in files:
                class_name = os.path.basename(root)
                if class_name not in category_dict2:
                    category_dict2[class_name] = category_count2
                    category_count2 += 1
                category = category_dict2[class_name]
                print(f'{file}')
                with open(f'{root}/{file}', 'r', encoding='utf-8') as inFile:
                    text = inFile.read()
                fdDict2['index'].append(len(fdDict2['index']))
                fdDict2['text'].append(text)
                fdDict2['class'].append(class_name)
                fdDict2['category'].append(category)
    return fdDict1, fdDict2


outDict1, outDict2 = collectFilesToDir('SSD_HDD_CPU')
df1 = pd.DataFrame(outDict1)
df1 = df1.reset_index(drop=True)
print(df1)
df2 = pd.DataFrame(outDict2)
df2 = df2.reset_index(drop=True)
print(df2)


# Функция для очистки текста
def clean_text(text):
    # приводим текст к нижнему регистру
    text = text.lower()
    # создаем регулярное выражение для удаления лишних символов
    regular = r'[\*+\#+\№\"\-+\+\=+\?+\&\^\.+\;\,+\>+\(\)\/+\:\\+]'
    # регулярное выражение для замены ссылки на "URL"
    regular_url = r'(http\S+)|(www\S+)|([\w\d]+www\S+)|([\w\d]+http\S+)'
    # удаляем лишние символы
    text = re.sub(regular, '', text)
    # заменяем ссылки на "URL"
    text = re.sub(regular_url, r'URL', text)
    # заменяем числа и цифры на ' пробел '
    text = re.sub(r'(\d+\s\d+)|(\d+)', ' ', text)
    # удаляем лишние пробелы
    text = re.sub(r'\s+', ' ', text)
    # возвращаем очищенные данные
    return text


# создаем список для хранения преобразованных данных
cleaned_text1 = []
cleaned_text2 = []
tokens1 = []
tokens2 = []
# загружаем стоп-слова
stop_words = stopwords.words('russian')
stop_words2 = stopwords.words('english')

# создаем список для хранения данных без стоп-слов
stopwords_cleaned1 = []
stopwords_cleaned2 = []
# инициализируем лемматайзер
lemmatizer = WordNetLemmatizer()

# для каждого сообщения text из столбца df1['text']
for text in df1['text']:
    # cleaning
    text = clean_text(text)

    # tokenization
    text = word_tokenize(text)
    tokens1.append(text)

    # удаление стоп-слов
    # с помощью цикла for поочередно проверяем каждое слово(токен) текущего сообщения
    # на вхождение в список стоп-слов. Если токен не является стоп-словом,
    # то добавляем его в список stopwords_cleaned
    text = [word for word in text if word not in stop_words]
    stopwords_cleaned1.append(text)
    text = [word for word in text if word not in stop_words2]
    stopwords_cleaned1.append(text)

    # лемматизация
    # text = [lemmatizer.lemmatize(w) for w in text]

# для каждого сообщения text из столбца df2['text']
for text in df2['text']:
    # cleaning
    text = clean_text(text)

    # tokenization
    text = word_tokenize(text)
    tokens2.append(text)

    # удаление стоп-слов
    # с помощью цикла for поочередно проверяем каждое слово(токен) текущего сообщения
    # на вхождение в список стоп-слов. Если токен не является стоп-словом,
    # то добавляем его в список stopwords_cleaned
    text = [word for word in text if word not in stop_words]
    stopwords_cleaned2.append(text)
    text = [word for word in text if word not in stop_words2]
    stopwords_cleaned2.append(text)

    # лемматизация
    # text = [lemmatizer.lemmatize(w) for w in text]

# Создание экземпляра класса MorphAnalyzer из pymorphy2
morph = pymorphy2.MorphAnalyzer()


# Функция для преобразования слова в начальную форму с использованием pymorphy
def lemmatize_word(word):
    return morph.parse(word)[0].normal_form


# Функция для преобразования текста в начальные формы слов с использованием pymorphy
def lemmatize_text(text):
    words = text.split()
    lemmatized_words = [lemmatize_word(word) for word in words]
    return ' '.join(lemmatized_words)


output_df_pymorphy1 = df1['text'].apply(lemmatize_text)
output_df_pymorphy2 = df2['text'].apply(lemmatize_text)

# добавляем преобразованный текст в список processed_text
cleaned_text1.append(text)
cleaned_text2.append(text)

# df['предобработанный_текст'] = cleaned_text
df1['предобработанный_текст'] = df1['text'].apply(lambda text: clean_text(text))
df2['предобработанный_текст'] = df2['text'].apply(lambda text: clean_text(text))

df1['cleaned_text'] = df1['предобработанный_текст'].apply(lemmatize_text)
df2['cleaned_text'] = df2['предобработанный_текст'].apply(lemmatize_text)

# df[cleaned_text].to_csv('clean_data.csv')
# print(df)
list_text1 = df1['cleaned_text'].tolist()
list_text2 = df2['cleaned_text'].tolist()
list_category1 = df1['category'].tolist()
list_category2 = df2['category'].tolist()
train_data = df1
test_data = df2


def button_callback():
    EPOCH_COUNT_BERT = int(entry_epoch_count.get())
    CLASS_LABELS = int(entry_class_labels.get())

    # https://huggingface.co/cointegrated/rubert-tiny ruBERT
    classifier = BertClassifier(
        model_path='cointegrated/rubert-tiny',
        tokenizer_path='cointegrated/rubert-tiny',
        n_classes=CLASS_LABELS,
        epochs=EPOCH_COUNT_BERT,
        model_save_path='bert.pt'
    )
    classifier.preparation(
        X_train=list(train_data['text']),
        y_train=list(train_data['category']),
        X_valid=list(test_data['text']),
        y_valid=list(test_data['category'])
    )
    classifier.train()

    texts = list(test_data['text'])
    labels = list(test_data['category'])
    predictions = [classifier.predict(t) for t in texts]

    precision, recall, f1score = precision_recall_fscore_support(labels, predictions, average='macro')[:3]
    print(f'precision: {precision:.3f}, recall: {recall:.3f}, f1score: {f1score:.3f}')
    text_area.insert('end', f'precision: {precision:.3f}, recall: {recall:.3f}, f1score: {f1score:.3f}\n')
    result_df = pd.DataFrame({
        'предсказанный_класс': predictions,
        'category': list(test_data['category'])
    })
    result_df = result_df.merge(df2[['category', 'class']], on='category', how='left')

    result_df = result_df.rename(columns={'class': 'реальный_класс'})

    df2_unique_index = df2.drop_duplicates('category').set_index('category')
    result_df['предсказанный_класс'] = result_df['предсказанный_класс'].map(df2_unique_index['class'])

    result_df = result_df.drop(columns=['category'])  # удалить столбец 'category'

    result_df.to_csv('classified_data.csv', index=False, sep=',', encoding='utf-8-sig')


def quit_button_callback():
    global root, entry_epoch_count, entry_class_labels

    def write_to_scrolledtext(msg):
        text_area.insert(tk.END, msg)
        text_area.see(tk.END)  # Автоматическая прокрутка текста вниз

    sys.stdout.write = write_to_scrolledtext  # Перенаправление sys.stdout в функцию write_to_scrolledtext

    root.title("Модель классификации текста")

    label_epoch_count = tk.Label(root, text="Количество эпох при обучении модели:")
    label_epoch_count.pack()

    entry_epoch_count = tk.Entry(root, textvariable=tk.StringVar(value="4"))
    entry_epoch_count.pack()

    label_class_labels = tk.Label(root, text="Предполагаемое количество классов:")
    label_class_labels.pack()

    entry_class_labels = tk.Entry(root, textvariable=tk.StringVar(value="3"))
    entry_class_labels.pack()

    run_button = tk.Button(root, text="Запустить обучение модели", command=button_callback)
    run_button.pack()

    quit_button = tk.Button(root, text="Завершить работу программы", command=root.destroy)
    quit_button.pack()


root = tk.Tk()


def create_text_area():
    global text_area
    text_area = scrolledtext.ScrolledText(root, height=10, width=100)  # use ScrolledText
    # text_area = tk.Text(root, height=10, width=50)
    text_area.pack()


create_text_area()
# quit_button = tk.Button(root, text="Завершить работу программы", command=quit_button_callback)
# quit_button.pack()
quit_button_callback()
root.mainloop()

