# %% パターン4でインストールしたライブラリの動作確認
import torch
import numpy as np
data = [[1, 2],[3, 4]]
x_data = torch.tensor(data)
np_array = np.array(data)
x_np = torch.from_numpy(np_array)
x_ones = torch.ones_like(x_data) # x_dataの特性（プロパティ）を維持
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float) # x_dataのdatatypeを上書き更新
print(f"Random Tensor: \n {x_rand} \n")
# %%
