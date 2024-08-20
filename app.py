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
    df = preprocess_data(df)

    print(df.columns)

    # Dividindo os dados e aplicando o pré-processamento
    target_columns = ["CPU Usage", "Memory Usage", "Thread Count", "Error Count",
                      "Total Data Received", "Total Data Filtered", "Total Data Compressed",
                      "Total Data Aggregated", "Total Data After Heuristics"]

    for target in target_columns:
        print(f"\nTreinando e avaliando para target: {target}")
        X_train, X_test, y_train, y_test = split_data(df, target)

        # Identificando as colunas numéricas e categóricas no X_train
        numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = X_train.select_dtypes(include=['object']).columns

        # Criando o pipeline de pré-processamento com base nas colunas presentes
        preprocessor = create_preprocessing_pipeline(numeric_features, categorical_features)

        # Aplicando o pré-processamento
        print(f"Aplicando pré-processamento para target: {target}")
        X_train = preprocessor.fit_transform(X_train)
        X_test = preprocessor.transform(X_test)

        # Treinando os modelos
        models = train_models(X_train, y_train, target)

        # Avaliando os modelos
        evaluate_models(models, X_test, y_test)
        print("\n" + "-" * 50 + "\n")


if __name__ == "__main__":
    main()
