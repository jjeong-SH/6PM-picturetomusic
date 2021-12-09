import os
import re
import streamlit as st
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import time
import base64
import json
import sklearn
import yt_dlp

from sklearn.preprocessing import MinMaxScaler
from googleapiclient.discovery import build
from datetime import datetime
from tqdm import tqdm
from PIL import Image
from torchvision.models import resnet18
from recommender import recommend, link_youtube



def documentation():
    st.subheader('''
    6PM documentation (사용설명서)
    ''')
    main_img = Image.open('images/main_image.png')
    st.image(main_img)
    document1 = Image.open('images/doc_slide_1.PNG')
    st.image(document1)
    document2 = Image.open('images/doc_slide_2.PNG')
    st.image(document2)
    st.write('''
    #### 1-1. Youtube API quota에 대한 추가 설명
    ''')
    st.write('''
    해당 에러**(googleapiclient.errors.HttpError)**가 뜬다면 
    요청할 수 있는 API 일일 할당량을 초과한 것이니 
    
    다시 할당량이 복구될 수 있도록 하루 기다려주시기 바랍니다ㅠㅠㅠㅠ
    ''')
    st.write("👇👇👇👇👇")
    q_img = Image.open('images/quota_exceederror.png')
    st.image(q_img)


@st.cache(allow_output_mutation=True)
def load_model():
    num_classes = 6
    model = resnet18()
    model.fc = nn.Linear(in_features=512, out_features=num_classes)
    return model
    

def music_downloader(url, name):
    connection = True
    with st.spinner("Wait while downloading..."):
        try:
            with yt_dlp.YoutubeDL({}) as ydl:
                ydl.download([url])
            time.sleep(5)
        except Exception as e:
            this = e
            connection = False
            return this, connection
        
        files = os.listdir('.')
        try:
            this_ = [file for file in files if name in file]
            this = this_[0]
        except:
            this = "<'ErrorBy6PMDevelopers'> : Denied by youtube downloader module. Click the above link to stream music"
            connection = False

    return this, connection


def music_stream(file):
    audio_bytes = open(file, "rb").read()
    st.audio(audio_bytes, format='audio/mp4')
    
    
def inference_start(img_tensor):
    with st.spinner("Wait while inferencing..."):
        placeholder = st.empty()
        spinner_file = open("images/Spinner-1s-200px.gif", "rb")
        contents = spinner_file.read()
        data_url = base64.b64encode(contents).decode("utf-8")
        spinner_file.close()
        placeholder.markdown(
            f'<p align="center"><img src="data:image/gif;base64,{data_url}" alt="spinner gif"></p>',
            unsafe_allow_html=True
        )
        # 여기에 inference
        # -----------------------------------------------------------------
        batch_tensor = torch.unsqueeze(img_tensor, 0)
        model = load_model()
        device = torch.device('cpu')
        model.load_state_dict(torch.load('exp1_best_model.pth', map_location=device))
        model.eval()
        output = model(batch_tensor)
        # st.write(output.shape)
        classes = ['anxiety', 'depression', 'joy', 'lonely', 'love', 'stress']
        _, index = torch.max(output, 1)
        percentage = nn.functional.softmax(output, dim=1)[0] * 100
        #st.write(classes[index[0]], percentage[index[0]].item())
        sentiment_ = classes[index[0]]
        st.write("Sentiment **{}** by probability of **{}**".format(sentiment_, percentage[index[0]].item()))
        # -----------------------------------------------------------------
        st.session_state.sentiment = sentiment_
        placeholder.empty()
            
    if not st.success("Done!"):
        st.stop()


# 캐시 삭제용 inference function (다른 위젯 클릭했는데 돌아가면 망한거임)
@st.cache(allow_output_mutation=True)
def inference(img_tensor):
    # 여기에 inference
    # -----------------------------------------------------------------
    batch_tensor = torch.unsqueeze(img_tensor, 0)
    model = load_model()
    device = torch.device('cpu')
    model.load_state_dict(torch.load('exp1_best_model.pth', map_location=device))
    model.eval()
    output = model(batch_tensor)
    # st.write(output.shape)
    classes = ['anxiety', 'depression', 'joy', 'lonely', 'love', 'stress']
    _, index = torch.max(output, 1)
    percentage = nn.functional.softmax(output, dim=1)[0] * 100
    #st.write(classes[index[0]], percentage[index[0]].item())
    sentiment_ = classes[index[0]]
    #st.write("Sentiment **{}** by probability of **{}**".format(sentiment_, percentage[index[0]].item()))
    perc_sent = percentage[index[0]].item()
    # -----------------------------------------------------------------
    st.session_state.sentiment = sentiment_
        
    return sentiment_, perc_sent


def feedback_after_program(image_pil, sentiment):
    # ==========================================================
    # 3. 피드백 남기기
    # ==========================================================
    st.subheader('''
    3. Leave Feedback
    ''')
    st.write('''
    #### (1) 사진에서 '{}' 분위기를 느낄 수 있었나요?
    '''.format(sentiment))
    st.image(image_pil)
    choice_1 = st.radio("Choose one for question <1>", ("YES", "NO"))
    if choice_1 == "NO":
        options = st.multiselect(
            "NO라고 대답하셨다면, 어떤 분위기를 느끼셨는지 골라주세요.",
            ['anxiety', 'depression', 'joy', 'lonely', 'love', 'stress']
        )
    st.write('''
    #### (2) 노래 추천은 만족스러웠나요?
    ''')
    choice_2 = st.radio("Choose one for question <2>", ("YES", "NO"))
    if choice_2 == "NO":
        opinion = st.selectbox(
            "어떤 점이 불만족스러웠나오?",
            ("장르에 맞는 노래가 아니어서", "분위기에 노래가 맞지 않아서", "좋아하는 노래가 아니어서", "기타")
        )
        if opinion == "기타":
            other_op = st.text_input("다른 의견이 있다면 써주세요!")
    emoji = '😘'
    if st.button("Send"):
        #st.write("Thank you for using our service!")
        st.markdown(f"<h4 style='text-align: center;'>{emoji} Thank you for using our service! {emoji}</h4>", unsafe_allow_html=True)
        st.balloons()
    
    
def run_program():
    # ==========================================================
    # 1. 이미지 업로드하기
    # ==========================================================
    st.subheader('''
    1. Search for a picture to upload
    ''')
    upload_img = st.file_uploader(" ")
    
    if not upload_img:
        st.stop()
        
    image_pil = Image.open(upload_img)
    st.image(image_pil)
    if "img_name" not in st.session_state:
        st.session_state.img_name = re.findall('name=\'(.*)\',\stype', str(upload_img))[0]
        st.session_state.uploaded_img = upload_img
        tf = transforms.ToTensor()
        img_tensor = tf(image_pil)
        img_tensor = transforms.Resize((256, 256))(img_tensor)
        inference_start(img_tensor)
    else:
        now = re.findall('name=\'(.*)\',\stype', str(upload_img))[0]
        if st.session_state.img_name != now:
            #st.write("사진 바뀌었음")
            st.session_state.img_name = now
            st.session_state.uploaded_img = upload_img
            tf = transforms.ToTensor()
            img_tensor = tf(image_pil)
            img_tensor = transforms.Resize((256, 256))(img_tensor)
            inference_start(img_tensor)
        else:
            #st.write("사진 그대로")
            st.session_state.uploaded_img = upload_img
            tf = transforms.ToTensor()
            img_tensor = tf(image_pil)
            img_tensor = transforms.Resize((256, 256))(img_tensor)
            with st.spinner("Wait while inferencing..."):
                placeholder = st.empty()
                spinner_file = open("images/Spinner-1s-200px.gif", "rb")
                contents = spinner_file.read()
                data_url = base64.b64encode(contents).decode("utf-8")
                spinner_file.close()
                placeholder.markdown(f'<p align="center"><img src="data:image/gif;base64,{data_url}" alt="spinner gif"></p>', unsafe_allow_html=True)
                sentiment_, perc_sent = inference(img_tensor)
                placeholder.empty()
                st.write("Sentiment **{}** by probability of **{}**".format(sentiment_, perc_sent))
            if not st.success("Done!"):
                st.stop()
            
    
    # ==========================================================
    # 2. 노래 추천받기
    # ==========================================================
    st.write('')
    st.subheader('''
    2. Get music recommendation
    ''')
    st.write(' ')
    genre = st.radio(
    "What genre do you want your music to be in?",
        ('POP', '발라드', 'R&B/Soul', '일렉트로니카', '포크', '록/메탈', '랩/힙합')
    )
    queries = recommend(st.session_state.sentiment, genre)
    
    if not st.checkbox("Submit"):
        st.error("Check the Submit box for recommendation")
        st.stop()

    st.write("")
    video_ids = []
    video_names = []
    for query in queries:
        video_id = link_youtube(query)['items'][0]['id']['videoId']
        link_url = f'https://www.youtube.com/watch?v={video_id}'
        video_name = link_youtube(query)['items'][0]['snippet']['title']
        name_ = re.sub(r'[|/]', '_', video_name)
        st.markdown(f"<h5 style='text-align: left;'>{video_name}</h5>", unsafe_allow_html=True)
        st.write("Link: {}".format(link_url))
        music_file, connection = music_downloader(link_url, name_)

        st.write("Music Streaming:")
        if connection:
            music_stream(music_file)
            os.remove(music_file)
        else:
            if type(music_file) == str:
                st.write("*{}*".format(music_file))
            else:
                st.write("*{} : {}*".format(type(music_file), music_file))
            
        st.write("")
        st.write("")
        
    ending_text = '''
    This is the end of recommendation service!
    '''
    st.markdown(f"<h4 style='text-align: center;'>{ending_text}</h4>", unsafe_allow_html=True)
    st.markdown(f"<h5 style='text-align: center;color: red;'>Make Sure you leave feedbacks by switching to [Feedback] Menu on the sidebar!</h5>", unsafe_allow_html=True)
    st.write("👈👈👈👈👈👈")
    st.write("👈👈👈👈👈👈")
        
        
def opinion_page():
    st.write('''
    #### Leave Free Opinions!
    ''')
    username_text = st.text_input("User Name")
    st.checkbox("Confirm")
    feedback_text = st.text_area("Opinion Area")
    if not username_text:
        st.warning("You MUST enter your User Name to leave a message")
        st.stop()
    else:
        st.button("Send")
        
        
def feedback_after_program():
    # ==========================================================
    # 3. 피드백 남기기
    # ==========================================================
    st.subheader('''
    3. Leave Feedback
    ''')
    st.write('''
    #### 1. 사진에서 '{}' 분위기를 느낄 수 있었나요?
    '''.format(st.session_state.sentiment))
    image_pil = Image.open(st.session_state.uploaded_img)
    st.image(image_pil)
    choice_1 = st.radio("Choose one for question <1>", ("YES", "NO"))
    if choice_1 == "NO":
        options = st.multiselect(
            "NO라고 대답하셨다면, 어떤 분위기를 느끼셨는지 골라주세요.",
            ['anxiety', 'depression', 'joy', 'lonely', 'love', 'stress']
        )
    st.write('''
    #### 2. 노래 추천은 만족스러웠나요?
    ''')
    choice_2 = st.radio("Choose one for question <2>", ("YES", "NO"))
    if choice_2 == "NO":
        opinion = st.selectbox(
            "어떤 점이 불만족스러웠나오?",
            ("장르에 맞는 노래가 아니어서", "분위기에 노래가 맞지 않아서", "좋아하는 노래가 아니어서", "기타")
        )
        if opinion == "기타":
            other_op = st.text_input("다른 의견이 있다면 써주세요!")
    emoji = '😘'
    if st.button("Send"):
        st.markdown(f"<h4 style='text-align: center;'>{emoji} Thank you for using our service! {emoji}</h4>", unsafe_allow_html=True)
        st.balloons()


def home_page():
    st.write('''
    #### Login by User Name
    ''')
    username = st.sidebar.text_input("User Name")
    if st.sidebar.checkbox("Confirm"):
        st.success("Logged In as {}".format(username))

        task = st.selectbox("Task", ["Documentation", "Run Program"])
        if task == "Documentation":
            documentation()

        elif task == "Run Program":
            run_program()
    else:
        st.warning("Please Enter Username to start Demopage")
        placeholder = st.empty()
        intro_img = open("images/introduction.png", "rb")
        contents = intro_img.read()
        img_url = base64.b64encode(contents).decode("utf-8")
        intro_img.close()
        placeholder.markdown(
            f'<p align="center"><img src="data:image/gif;base64,{img_url}" alt="spinner gif"></p>',
            unsafe_allow_html=True
        )
        
        

if __name__ == "__main__":
    st.set_page_config(layout="wide")
    showWarningOnDirectExecution = False
    
    st.markdown("""
        <style>
        .css-1aumxhk {
            padding: 0em 1em;
        }
        </style>
    """, unsafe_allow_html=True)
    st.sidebar.markdown(f"<h5 style='text-align: right;'>Developed by 빅데이터종합설계 2조</h5>", unsafe_allow_html=True)
    st.title("6PM - Picture to Music Web Service")
    st.write("you can check our source code via github [here](https://github.com/jjeong-SH/6PM-picturetomusic)!")
    st.sidebar.title("Sidebar")
    menu = ["Home", "Feedback", "Opinions"]
    choice = st.sidebar.selectbox("Menu", menu)
    if choice == "Home":
        home_page()
    elif choice == "Feedback":
        feedback_after_program()
    else:
        opinion_page()
