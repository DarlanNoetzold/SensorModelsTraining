from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from joblib import dump


def train_models(X_train, y_train):
    models = {
        "RandomForest": RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_split=5, min_samples_leaf=3,
                                              random_state=42),
        "XGBoost": XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=8, subsample=0.8, colsample_bytree=0.8,
                                random_state=42),
        "LightGBM": LGBMRegressor(n_estimators=200, learning_rate=0.05, max_depth=8, subsample=0.8,
                                  colsample_bytree=0.8, random_state=42),
        "CatBoost": CatBoostRegressor(iterations=300, learning_rate=0.03, depth=10, l2_leaf_reg=3, random_state=42,
                                      verbose=0),
        "LinearRegression": LinearRegression(fit_intercept=True, normalize=True),
        "SVR": SVR(C=1.0, epsilon=0.1, kernel='rbf', gamma='scale'),
        "KNN": KNeighborsRegressor(n_neighbors=5, weights='distance', algorithm='auto'),
        "MLP": MLPRegressor(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', max_iter=1000,
                            random_state=42)
    }

    best_models = {}

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        best_models[name] = model
        # Exportando o modelo após o treinamento
        dump(model, f'/app/models/{name}_model.joblib')
        print(f'{name} model saved.')

    return best_models
