import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PowerTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression
import os


def load_data():
    folder_path = "/app/data/"

    csv_files = [f for f in os.listdir(folder_path) if f.startswith("metrics_") and f.endswith(".csv")]

    if not csv_files:
        raise FileNotFoundError("Nenhum arquivo CSV encontrado na pasta.")

    csv_files.sort(reverse=True)

    latest_csv_file = csv_files[0]

    df = pd.read_csv(os.path.join(folder_path, latest_csv_file))

    return df


def preprocess_data(df):
    # Tratamento de dados faltantes
    df = df.dropna()

    # Tratamento de outliers com transformação robusta
    transformer = PowerTransformer()
    df[df.select_dtypes(include=['float64', 'int64']).columns] = transformer.fit_transform(
        df.select_dtypes(include=['float64', 'int64'])
    )

    return df


def create_preprocessing_pipeline(numeric_features, categorical_features):
    # Normalização e vetorizaçao das colunas numéricas
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Vetorização das colunas categóricas
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Aplica transformação para numéricas e categóricas
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    return preprocessor

def apply_feature_selection(X_train, y_train):
    # Aplicação de PCA e seleção de características após o pré-processamento
    feature_selector = Pipeline(steps=[
        ('pca', PCA(n_components=0.95)),  # Mantendo 95% da variância
        ('feature_selection', SelectKBest(score_func=f_regression, k=10))
    ])

    X_train_selected = feature_selector.fit_transform(X_train, y_train)
    return feature_selector, X_train_selected


def split_data(df, target_column):
    print(f"Tentando separar o target: {target_column}")

    if target_column not in df.columns:
        raise ValueError(
            f"A coluna {target_column} não foi encontrada no DataFrame. Colunas disponíveis: {df.columns.tolist()}")

    X = df.drop(columns=[target_column])
    y = df[target_column]

    print(f"Colunas em X após o drop: {X.columns.tolist()}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"Shape de X_train: {X_train.shape}")
    print(f"Shape de y_train: {y_train.shape}")

    return X_train, X_test, y_train, y_test