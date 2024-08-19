import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PowerTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression


def load_data():
    return pd.read_csv("/app/shared/metrics.csv")


def preprocess_data(df):
    # Tratamento de dados faltantes
    df = df.dropna()

    # Tratamento de outliers com transformação robusta
    transformer = PowerTransformer()
    df[df.select_dtypes(include=['float64', 'int64']).columns] = transformer.fit_transform(
        df.select_dtypes(include=['float64', 'int64'])
    )

    return df


def create_preprocessing_pipeline(df):
    # Normalização e vetorizaçao das colunas numéricas
    numeric_features = df.select_dtypes(include=['int64', 'float64']).columns
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=0.95)),  # Mantendo 95% da variância
        ('feature_selection', SelectKBest(score_func=f_regression, k=10))
    ])

    # Vetorização das colunas categóricas
    categorical_features = df.select_dtypes(include=['object']).columns
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    return preprocessor


def split_data(df, target_column):
    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test
    