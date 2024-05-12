import pandas as pd
import numpy as np
import re
import spacy
import unicodedata
from natasha import Doc, Segmenter, MorphVocab, NewsEmbedding, NewsMorphTagger

class TextNormalizer:
    def __init__(self, name_keys=None):
        # Инициализация Natasha для русского языка
        self.segmenter = Segmenter()
        self.morph_vocab = MorphVocab()
        self.emb = NewsEmbedding()
        self.morph_tagger = NewsMorphTagger(self.emb)

        # Инициализация spaCy для английского языка
        self.nlp_en = spacy.load("en_core_web_sm")

        # Инициализация списка исключений, если он предоставлен
        self.name_keys = name_keys if name_keys is not None else []

    def clean_text(self, text):
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'[^\w\s]', ' ', text)
        for word in self.name_keys:
            text = re.sub(r'\b' + word + r'\b', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        return text

    def normalize_text(self, text, lang='ru'):
        if lang == 'ru':
            doc = Doc(text)
            doc.segment(self.segmenter)
            doc.tag_morph(self.morph_tagger)
            for token in doc.tokens:
                token.lemmatize(self.morph_vocab)
            return ' '.join([_.lemma for _ in doc.tokens if _.lemma.isalpha()])
        elif lang == 'en':
            doc = self.nlp_en(text)
            return ' '.join([token.lemma_ for token in doc if token.is_alpha])

    def detect_language(self, text):
        for char in text:
            if unicodedata.category(char).startswith('L') and unicodedata.name(char).startswith('CYRILLIC'):
                return 'ru'
        return 'en'

    def normalize_df(self, df, input_columns, output_column=None):
        if output_column is None:
            output_column = input_columns

        for in_col, out_col in zip(input_columns, output_column):
            df[out_col] = df[in_col].apply(lambda x: self.clean_text(str(x)))
            lang = self.detect_language(df[out_col].str.cat())
            df[out_col] = df[out_col].apply(lambda x: self.normalize_text(x, lang=lang))
        return df

def prepare_male_female_dataset(test_df):
    test_df = test_df.reset_index()
    test_df.rename(columns={'index': 'old_index'}, inplace=True)

    clothing_categories = [
        'Одежда', 'Спортивная одежда', 'Обувь', 'Спортивная обувь',
        'Головные уборы', 'Белье', 'Аксессуары', 'Бижутерия', 'Ювелирные украшения'
    ]

    clothes = test_df[test_df['category'].isin(clothing_categories)]

    normalizer = TextNormalizer()

    clothes = normalizer.normalize_df(
        clothes, ['description'], ['clear_description'])

    clothes['clear_description'] = clothes['clear_description'].str.split()

    men_related = ['папа', 'брат', 'дедушка', 'дядя', 'жених',
                   'парень', 'муж', 'мужской', 'мужчина']

    women_related = ['юбка', 'платье', 'блузка', 'бюстгальтер', 'леггинс',
                     'колготки', 'туника', 'сестра', 'бабушка', 'тетя',
                     'девушка', 'женский', 'женщина']

    def contains_word(word_list, gender):
        return bool(set(gender) & set(word_list))

    filtered_male = clothes[clothes['clear_description'].apply(
        lambda x: contains_word(x, gender=men_related))]

    filtered_female = clothes[clothes['clear_description'].apply(
        lambda x: contains_word(x, gender=women_related))]

    male_indices = set(filtered_male.index)
    female_indices = set(filtered_female.index)

    male_unique_indices = male_indices - female_indices
    female_unique_indices = female_indices - male_indices

    male_clothes = clothes.loc[list(male_unique_indices)]
    female_clothes = clothes.loc[list(female_unique_indices)]

    targets = pd.Series(index=clothes.index)

    targets.loc[female_clothes.index] = 0
    targets.loc[male_clothes.index] = 1
    targets.dropna(inplace=True)

    # Создаем датафрейм с target и index
    male_female_df = pd.DataFrame({
        'index': targets.index,
        'target': targets.values.astype(int)
    })

    return male_female_df

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare Male/Female dataset")
    parser.add_argument('--input_file', type=str, required=True, help='Path to the test dataframe file')
    parser.add_argument('--output_file', type=str, required=True, help='Path to save the resulting CSV file')

    args = parser.parse_args()

    # Загрузка тестового DataFrame
    test_df = pd.read_csv(args.input_file)

    # Подготовка данных
    result_df = prepare_male_female_dataset(test_df)

    # Сохранение результата
    result_df.to_csv(args.output_file, index=False)
