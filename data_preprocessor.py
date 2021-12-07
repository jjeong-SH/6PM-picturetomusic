from datetime import datetime
from tqdm import tqdm
import pandas as pd
import numpy as np
import json
import os
import sklearn
from sklearn.preprocessing import MinMaxScaler

import warnings

warnings.filterwarnings('ignore')


def load_raw_data(filepath=None):

    sentiment_list = ['lonely','joy','joy',
                      'anxiety', 'love', 'love',
                      'stress', 'depression','love',
                      'depression', 'lonely', 'stress','joy','joy']
    music_data = pd.DataFrame()
    singer = []
    release = []
    genre = []
    content = []
    likes = []
    sentiment = []
    if not filepath:
        filepath = './gdrive/Shareddrives/빅종설/음원 크롤링/json파일'
    # music json data
    for i, filename in tqdm(enumerate(os.listdir(filepath))):
        with open(os.path.join(filepath, filename), "r") as f:
            file = json.load(f)
            for record in tqdm(file):
                singer.append(record['가수'])
                release.append(record['발매일'])
                genre.append(record['장르'])
                content.append(record['제목'])
                likes.append(record['좋아요 수'])
                sentiment.append(sentiment_list[i])

    music_data['singer'] = singer
    music_data['release'] = release
    music_data['genre'] = genre
    music_data['content'] = content
    music_data['likes'] = likes
    music_data['sentiment'] = sentiment

    return music_data


def fill_date(row):
    if len(row) == 7:  # 일자가 없는 경우
        row += '.01'  # 1일로 채워준다
    elif len(row) == 4:  # 월, 일자가 없는 경우
        row += '.01.01'  # 1월 1일로 채워준다
    else:
        pass
    return row


def feature_preprocess(music_data):
    music_data['release_len'] = music_data['release'].apply(lambda row: True if len(row) < 9 else False)
    music_data['release_filled'] = music_data['release'].apply(lambda row: fill_date(row))
    music_data = music_data[music_data['release_filled'] != '-']  # 날짜 정보가 아예 없는 경우 제외
    music_data.drop(columns=['release', 'release_len'], inplace=True)
    music_data['release_filled'] = music_data['release_filled'].apply(
        lambda row: datetime.strptime(row, '%Y.%m.%d').strftime('%Y-%m-%d'))

    music_data['likes'] = music_data['likes'].apply(lambda row: row.replace(',', ''))
    music_data['likes'] = music_data['likes'].astype(int)
    return music_data


def engineer_feature(music_data):
    music_data['release_rank'] = music_data['release_filled'].rank()
    music_data['release_rank'] = MinMaxScaler().fit_transform(music_data['release_rank'].to_numpy().reshape(-1, 1))

    music_data['year'] = music_data['release_filled'].apply(lambda row: row.split('-')[0])
    # 연단위로 MinMaxScale 함 -> 연단위로 급증하는 좋아요 수의 영향을 제거하기 위해
    music_data['scaled_likes'] = np.nan
    for y in music_data['year'].unique():
        likes_data = music_data['likes'][music_data['year'] == y]
        likes_data_idx = likes_data.index
        res = MinMaxScaler().fit_transform(likes_data.to_numpy().reshape(-1, 1))
        res = res.reshape(res.shape[0], )
        music_data['scaled_likes'].loc[likes_data_idx] = res

    music_data['feature'] = music_data['scaled_likes'] + music_data['release_rank']
    return music_data


def save_file(file_name, file):
    file.to_csv(f'./dataset/{file_name}.csv')
    print('music meta data saved')


def run_preprocessor():
    music_data = load_raw_data(filepath='./dataset/json_files')
    music_data = feature_preprocess(music_data)
    music_data = engineer_feature(music_data)
    save_file('music_meta_data', music_data)
    print('music_data preprocessed')
