from data_preprocessor import run_preprocessor
from googleapiclient.discovery import build

from datetime import datetime
from tqdm import tqdm
import pandas as pd
import numpy as np
import json
import os

import warnings

warnings.filterwarnings('ignore')


def recommend(sentiment, genre):
    file_path = 'dataset/music_meta_data.csv'
    music_data = pd.read_csv(file_path)

    # 원하는 감성, 장르에 해당하는 후보군만 뽑음
    candidates = music_data[(music_data['sentiment'] == sentiment) & (music_data['genre'].isin([genre]))]
    # feature 를 sampling probability로해서 후보군에서 3개의 노래를 샘플링함
    probs = candidates['feature'] / candidates['feature'].sum()
    probs = np.array(probs)
    rec_idx = np.random.choice(len(candidates), 3, replace=False, p=probs)
    result = candidates[['content', 'singer']].iloc[rec_idx]
    # result = candidates[['content', 'singer', 'sentiment', 'genre', 'release_filled', 'likes']].iloc[rec_idx]

    queries = []
    for content, singer in zip(result['content'].values, result['singer'].values):
        queries.append(f'{content} {singer}')

    return queries


def link_youtube(query):
    DEVELOPER_KEY = 'AIzaSyDqjhBRb96rl047foftBv-ytRXUmMRNMjk'
    YOUTUBE_API_SERVICE_NAME = "youtube"
    YOUTUBE_API_VERSION = "v3"
    youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=DEVELOPER_KEY)

    search_response = youtube.search().list(q=query, order="relevance",
                                            part="snippet", maxResults=1).execute()
    return search_response

'''
if __name__ == '__main__':
    run_preprocessor()
    sentiment = 'lonely'  # model output
    genre = '발라드'
    queries = recommend(sentiment, genre)
    for query in queries:
        res = link_youtube(query)
        print(res)
        break
'''
