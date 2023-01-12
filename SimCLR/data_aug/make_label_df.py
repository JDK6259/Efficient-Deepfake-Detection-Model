#%%
import os
import json
import numpy as np
import pandas as pd
from natsort import natsorted
from sklearn.model_selection import train_test_split

base_path =  'C:\\Users\\soyeon\\Desktop\\인지응\\project\\code'
# %%
######## Celeb-DF-v2
# test video list
test_df = pd.read_table(f'{base_path}\\cropped_images\\Celeb-DF-v2\\List_of_testing_videos.txt', sep=' ')
new_row = test_df.columns.values.tolist()
test_df.columns = ['label', 'files']
test_df = test_df.append({'label': new_row[0], 'files':new_row[1]}, ignore_index=True)
test_df['files'] = test_df['files'].apply(lambda x: '\\'.join(x[:-4].split('/')))

rows = []
for path, _, files in os.walk(f'{base_path}\\cropped_images\\Celeb-DF-v2'):
    if len(files)!=0:
      for file in files:
        if path.split('\\')[-2] == 'Celeb-real' or  path.split('\\')[-2] == 'YouTube-real':
          real_fake = 1
        else:
          real_fake = 0
        if '\\'.join(path.split('\\')[-2:]) in test_df.files.values:
          split = 'test'
        else : 
          split = 'train'
        rows.append((path.split('\\')[-1], path + '\\' + file, real_fake, split))
label_df = pd.DataFrame(rows, columns=['image_id', 'path', 'real_fake', 'split'])
label_df.to_csv(f'{base_path}\\cropped_images\\celeb_path.csv', index=False)
# %%
######## DFDC
with open(f'{base_path}\\cropped_images\\DFDC\\metadata.json') as json_file:
  json_data = json.load(json_file)

# 1) meta data -> image_id, label, split
df = []
for file_name in json_data:
    df.append((file_name[:-4], json_data[file_name]['label'], json_data[file_name]['split']))
df = pd.DataFrame(df, columns=['image_id', 'real_fake', 'split'])
df['real_fake'] = df['real_fake'].apply(lambda x: 0 if x=='FAKE' else 1)

# 2) meta data -> image_id, path
rows = []
for path, _, files in os.walk(base_path+'\\cropped_images\\DFDC'):
    if '.json' in files[0]:
        continue
    if len(files)!=0:
      for file in files:
        rows.append((path.split('\\')[-1], path + '\\' + file))
df2 = pd.DataFrame(rows, columns=['image_id', 'path'])

# 3) merge
dfdc = pd.merge(df, df2, on='image_id')

dfdc = dfdc.query('real_fake == 0')
train, test = train_test_split(dfdc, test_size=0.1, random_state=42)
dfdc.loc[train.index, 'split'] = 'train'
dfdc.loc[test.index, 'split'] = 'test'

dfdc.to_csv(f'{base_path}\\cropped_images\\dfdc_path.csv', index=False)
# %%
######## FF++
rows = []
df = []
for path, _, files in os.walk(base_path+'\\cropped_images\\FaceForensics++'):
    if len(files)!=0:
      for file in files:
        if path.split('\\')[-5] == 'manipulated_sequences':
          real_fake = 0
        else:
          real_fake = 1
        rows.append((path.split('\\')[-4], path.split('\\')[-1], path + '\\' + file, real_fake))
ff = pd.DataFrame(rows, columns=['cat', 'image_id', 'path', 'real_fake'])

# train / test split
sub = ff.query('cat != "youtube"')
train, test = train_test_split(sub, test_size=0.1, shuffle=True, stratify=sub['cat'], random_state=42)
ff.loc[train.index, 'split'] = 'train'
ff.loc[test.index, 'split'] = 'test'

ff.to_csv(f'{base_path}\\cropped_images\\ff++_path.csv', index=False)
# %%
ff = pd.read_csv(f'{base_path}\\cropped_images\\ff++_path.csv')
dfdc = pd.read_csv(f'{base_path}\\cropped_images\\dfdc_path.csv')
celeb = pd.read_csv(f'{base_path}\\cropped_images\\celeb_path.csv')

# %%
def make_len_50(df, columns, data_cat):
  gr_df = df.groupby(columns)[['path', 'real_fake', 'split']].apply(lambda x: x[::int(len(x)/50)][:50] if len(x)>=50 else print('pass'))
  gr_df = gr_df.reset_index()[['image_id', 'path', 'real_fake', 'split']]
  gr_df['category'] = data_cat
  return gr_df

df_ff = make_len_50(ff, ['cat','image_id'], 'ff++')
df_celeb = make_len_50(celeb, ['image_id'], 'celeb')
df_dfdc = make_len_50(dfdc, ['image_id'], 'dfdc')
# %%
all_df = pd.concat([df_ff, df_celeb, df_dfdc])
all_df.to_csv(f'{base_path}\\cropped_images\\same_frames_video.csv', index=False)
# %%
