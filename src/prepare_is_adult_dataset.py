import pandas as pd

def prepare_is_adult_dataset(test_df):
    # Создаем новый DataFrame с нужными колонками 'index' и 'target'
    is_adult_df = pd.DataFrame({
        'index': test_df.index,
        'target': test_df['isadult'].astype(int)  # Преобразуем True/False в 1/0
    })
    
    return is_adult_df

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare IsAdult dataset")
    parser.add_argument('--input_file', type=str, required=True, help='Path to the test dataframe file')
    parser.add_argument('--output_file', type=str, required=True, help='Path to save the resulting CSV file')

    args = parser.parse_args()

    # Загрузка тестового DataFrame
    test_df = pd.read_csv(args.input_file)

    # Подготовка данных
    result_df = prepare_is_adult_dataset(test_df)

    # Сохранение результата
    result_df.to_csv(args.output_file, index=False)