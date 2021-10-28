# %% パターン1でインストールしたライブラリの動作確認
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.svm import SVR
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from shapely.geometry import Point
# numpy
a = [1, 2, 3, 4]
avg = np.mean(a)
print(f'numpy {avg}')

# pandas
b = [5, 6, 7, 8]
df = pd.DataFrame([a, b])
print(f'pandas {df.mean(axis=1)}')

# scipy
c = [-3, -2, -1, 0, 1, 2, 3]
norm = stats.norm.pdf(c, loc=0, scale=1)
print(f'scipy {norm}')

# scikit-learn
estimator = SVR()
estimator.fit(np.array(c).reshape(-1, 1), np.array(norm))
pred = estimator.predict(np.array(c).reshape(-1, 1))
print(f'scikit-learn {pred}')

# lightgbm
estimator = LGBMRegressor()
estimator.fit(np.array(c).reshape(-1, 1), np.array(norm))
pred_lgbm = estimator.predict(np.array(c).reshape(-1, 1))
print(f'lightgbm {pred_lgbm}')

# xgboost
estimator = XGBRegressor()
estimator.fit(np.array(c).reshape(-1, 1), np.array(norm))
pred_xgb = estimator.predict(np.array(c).reshape(-1, 1))
print(f'xgboost {pred_xgb}')

# matplotlib
plt.plot(c, norm)
plt.show()

# seaborn
sns.scatterplot(c, norm)
sns.scatterplot(c, pred, color='red')
plt.show()

# geopandas
gdf_new = gpd.GeoDataFrame(crs = 'epsg:3099')
gdf_new['geometry'] = None
gdf_new.at[0, 'geometry'] = Point(0, 0)
gdf_new.at[1, 'geometry'] = Point(1, 1)
print(f'geopandas {gdf_new}')

# %%
