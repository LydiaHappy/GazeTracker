import json
import os
import pandas as pd
from io import StringIO
from collections import defaultdict
from tqdm.auto import tqdm


DATA_ROOT = 'gazecapture'

def gather(ignore_unprocessed=False):
    device_to_index = defaultdict(list)

    folders = os.listdir(DATA_ROOT)
    for folder_name in folders:
        folder_path = os.path.join(DATA_ROOT, folder_name)
        if not os.path.isdir(folder_path):
            continue
        if ignore_unprocessed and not os.path.exists(os.path.join(folder_path, 'out')):
            continue
        info_path = os.path.join(folder_path, 'info.json')
        if os.path.exists(info_path):
            with open(info_path) as f:
                j = json.load(f)
                device_to_index[j['DeviceName']].append(folder_name)
        else:
            print(info_path, 'not exists')
    return device_to_index


def stats(d):
    print(sorted([(k, len(d[k])) for k in d.keys()], key=lambda x: x[1]))


device_to_index = gather()
stats(device_to_index)

devices = device_to_index.keys()


def get_data(folder_name):
    folder_path = os.path.join(DATA_ROOT, folder_name)
    frames_path = os.path.join(folder_path, 'frames')
    frames = [f.replace('.jpg', '') for f in os.listdir(frames_path)]

    def read_json(name):
        path = os.path.join(folder_path, name)
        with open(path) as f:
            return json.load(f)

    # info
    info_json = read_json('info.json')
    screen_json = read_json('screen.json')
    dot_json = read_json('dotInfo.json')

    # merge feature csvs into one
    csv_header = None
    csv_body = []
    for i, frame in enumerate(frames):
        csv_path = os.path.join(folder_path, 'out', f'{frame}.csv')
        if os.path.exists(csv_path):
            with open(csv_path) as f:
                lines = f.readlines()
                if csv_header is None:
                    csv_header = lines[0]
                csv_body.append(lines[1])
        elif csv_header is not None:
            csv_body.append(','.join(['null'] * (csv_header.count(',') + 1)))

    # something is broken, ignore
    if csv_header is None:
        print(f'Error processing {csv_path}')
        return None

    # build data frame
    csv_string = '\n'.join([csv_header] + csv_body)
    feature_df = pd.read_csv(StringIO(csv_string), skipinitialspace=True)

    try:
        # merge index
        feature_df['folder'] = folder_name

        # merge screen info
        feature_df['H'] = screen_json['H']
        feature_df['W'] = screen_json['W']
        feature_df['Orientation'] = screen_json['Orientation']

        # merge device info
        feature_df['DeviceName'] = info_json['DeviceName']

        # merge train/test
        feature_df['Train'] = 1 if info_json['Dataset'] == 'train' else 0

        # merge dot info
        feature_df['XCam'] = dot_json['XCam']
        feature_df['YCam'] = dot_json['YCam']

    except:
        print(f'Error processing {folder_name}')

    return feature_df


df_all_list = []
for idx in tqdm(os.listdir(DATA_ROOT)):
    if os.path.exists(os.path.join(DATA_ROOT, idx, 'out')):
        if (df := get_data(idx)) is not None:
            df_all_list.append(df)

df_all = pd.concat(df_all_list, ignore_index=True)
df_all.to_csv('dataset_all.csv.gz', index=False)
print('df_all generated')

df_slim = pd.DataFrame()
eye_lmk_0_index = range(20, 28)
eye_lmk_1_index = range(48, 56)
df_slim['eye_lmk_0_X'] = df_all[[
    f'eye_lmk_X_{i}' for i in eye_lmk_0_index]].mean(axis=1)
df_slim['eye_lmk_0_Y'] = df_all[[
    f'eye_lmk_Y_{i}' for i in eye_lmk_0_index]].mean(axis=1)
df_slim['eye_lmk_0_Z'] = df_all[[
    f'eye_lmk_Z_{i}' for i in eye_lmk_0_index]].mean(axis=1)
df_slim['eye_lmk_1_X'] = df_all[[
    f'eye_lmk_X_{i}' for i in eye_lmk_1_index]].mean(axis=1)
df_slim['eye_lmk_1_Y'] = df_all[[
    f'eye_lmk_Y_{i}' for i in eye_lmk_1_index]].mean(axis=1)
df_slim['eye_lmk_1_Z'] = df_all[[
    f'eye_lmk_Z_{i}' for i in eye_lmk_1_index]].mean(axis=1)
columns = ['gaze_0_x', 'gaze_0_y', 'gaze_0_z',
           'gaze_1_x', 'gaze_1_y', 'gaze_1_z',
           'gaze_angle_x', 'gaze_angle_y',
           'H', 'W', 'Orientation', 'DeviceName',
           'Train', 'XCam', 'YCam']
for c in columns:
    df_slim[c] = df_all[c]
df_slim.to_csv('dataset.csv.gz', index=False)
print('df_slim generated')
