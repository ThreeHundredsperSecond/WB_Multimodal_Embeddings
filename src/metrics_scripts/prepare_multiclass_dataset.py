import pandas as pd
from sklearn.preprocessing import LabelEncoder

def replace_subcategory(df):
    # Создание копии DataFrame
    df_modified = df.copy()

    # Группировка по sub_category и подсчет количества строк в каждой группе
    counts = df_modified.groupby('sub_category').size()

    # Получение списка sub_category, где количество строк меньше 11
    sub_categories_to_replace = counts[counts < 11].index.tolist()

    # Замена sub_category на category для подходящих групп
    for sub_category in sub_categories_to_replace:
        category_value = df_modified.loc[df_modified['sub_category'] == sub_category, 'category'].iloc[0]
        df_modified.loc[df_modified['sub_category'] == sub_category, 'sub_category'] = category_value

    return df_modified

def prepare_multiclass_dataset(test_df):
    # Замена sub_category на основе их встречаемости
    sub_df = replace_subcategory(test_df)

    # Создание экземпляра LabelEncoder
    label_encoder = LabelEncoder()

    # Преобразование категориальных меток в числовые
    y_sub = label_encoder.fit_transform(sub_df['sub_category'])

    # Создаем DataFrame для сохранения
    multiclass_df = pd.DataFrame({
        'index': sub_df.index,
        'target': y_sub
    })

    return multiclass_df

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare Multiclass dataset")
    parser.add_argument('--input_file', type=str, required=True, help='Path to the test dataframe file')
    parser.add_argument('--output_file', type=str, required=True, help='Path to save the resulting CSV file')

    args = parser.parse_args()

    # Загрузка тестового DataFrame
    test_df = pd.read_csv(args.input_file)

    # Подготовка данных
    result_df = prepare_multiclass_dataset(test_df)

    # Сохранение результата
    result_df.to_csv(args.output_file, index=False)
