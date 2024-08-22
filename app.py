import os
from data_preprocessing import load_data, preprocess_data, create_preprocessing_pipeline, split_data
from model_training import train_models
from model_evaluation import evaluate_models

def remove_sequential_duplicates(df, ignore_columns):
    """
    Remove linhas consecutivas que têm os mesmos valores em todas as colunas,
    exceto nas colunas especificadas em ignore_columns.
    """
    df_subset = df.drop(columns=ignore_columns)

    df_shifted = df_subset.shift(1)

    mask = (df_subset != df_shifted).any(axis=1)

    df_cleaned = df[mask]

    return df_cleaned

def main():
    os.makedirs('models', exist_ok=True)

    df = load_data()
    print(df.columns)

    ignore_columns = ["ID", "CPU Usage", "Memory Usage", "Thread Count", "Error Count"]

    df = remove_sequential_duplicates(df, ignore_columns)

    df = preprocess_data(df)

    print(df.columns)

    # Dividindo os dados e aplicando o pré-processamento
    target_columns = ["CPU Usage", "Memory Usage", "Thread Count", "Error Count",
                      "Total Data Received", "Total Data Filtered", "Total Data Compressed",
                      "Total Data Aggregated", "Total Data After Heuristics"]

    for target in target_columns:
        print(f"\nTreinando e avaliando para target: {target}")
        X_train, X_test, y_train, y_test = split_data(df, target)

        numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = X_train.select_dtypes(include=['object']).columns

        preprocessor = create_preprocessing_pipeline(numeric_features, categorical_features)

        print(f"Aplicando pré-processamento para target: {target}")
        X_train = preprocessor.fit_transform(X_train)
        X_test = preprocessor.transform(X_test)

        models = train_models(X_train, y_train, target)

        evaluate_models(models, X_test, y_test)
        print("\n" + "-" * 50 + "\n")


if __name__ == "__main__":
    main()