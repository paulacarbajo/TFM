import pandas as pd

store = pd.HDFStore('data/processed/assets.h5', 'r')
print('Available keys:', store.keys())
key = store.keys()[0]
print(f'Using key: {key}')
df = store[key]
print('Columns:', df.columns.tolist())
print('ret_1d_forward exists:', 'ret_1d_forward' in df.columns)
print('Shape:', df.shape)
store.close()
