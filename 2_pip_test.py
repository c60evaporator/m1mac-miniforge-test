# %% パターン2でインストールしたライブラリの動作確認
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, KFold
import seaborn as sns
import optuna

# optuna
iris = sns.load_dataset("iris")
import numpy as np
X = iris[['petal_width', 'petal_length']].values  # 説明変数をndarray化
y = iris['species']  # 目的変数をndarray化
model = SVC()
scoring = 'f1_micro'  # 評価指標をf1_microに指定
seed = 42
cv = KFold(n_splits=3, shuffle=True, random_state=seed)

def bayes_objective(trial):
    params = {
        "gamma": trial.suggest_float("gamma", 0.001, 100, log=True),
        "C": trial.suggest_float("C", 0.01, 100, log=True)
    }
    # モデルにパラメータ適用
    model.set_params(**params)
    # cross_val_scoreでクロスバリデーション
    scores = cross_val_score(model, X, y, cv=cv,
                             scoring=scoring, n_jobs=-1)
    val = scores.mean()
    return val


study = optuna.create_study(direction="maximize",
                            sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(bayes_objective, n_trials=40)

# 最適パラメータの表示と保持
best_params = study.best_trial.params
best_score = study.best_trial.value
print(f'最適パラメータ {best_params}\nスコア {best_score}')
# %%
