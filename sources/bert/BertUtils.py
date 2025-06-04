import re


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

# Функция для преобразования слова в начальную форму с использованием pymorphy
def lemmatize_word(word, morph):
    return morph.parse(word)[0].normal_form

# Функция для преобразования текста в начальные формы слов с использованием pymorphy
def lemmatize_text(text, morph):
    words = text.split()
    lemmatized_words = [lemmatize_word(word, morph) for word in words]
    return ' '.join(lemmatized_words)

