import os
from data_preprocessing import load_data, preprocess_data, create_preprocessing_pipeline, split_data
from model_training import train_models
from model_evaluation import evaluate_models


def main():
    os.makedirs('models', exist_ok=True)

    # Carregando os dados
    df = load_data()
    print(df.columns)
    # Pré-processamento dos dados
    #df = preprocess_data(df)

    print(df.columns)

    # Criando o pipeline de pré-processamento
    preprocessor = create_preprocessing_pipeline(df)

    # Dividindo os dados e aplicando o pré-processamento
    target_columns = ["CPU Usage", "Memory Usage", "Thread Count", "Error Count",
                      "Total Data Received", "Total Data Filtered", "Total Data Compressed",
                      "Total Data Aggregated", "Total Data After Heuristics"]

    for target in target_columns:
        print(f"\nTraining and evaluating for target: {target}")
        X_train, X_test, y_train, y_test = split_data(df, target)

        # Aplicando o pré-processamento
        X_train = preprocessor.fit_transform(X_train)
        X_test = preprocessor.transform(X_test)

        # Treinando os modelos
        models = train_models(X_train, y_train)

        # Avaliando os modelos
        evaluate_models(models, X_test, y_test)
        print("\n" + "-" * 50 + "\n")


if __name__ == "__main__":
    main()
