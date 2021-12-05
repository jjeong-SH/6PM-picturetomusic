import streamlit as st
import numpy as np
import pandas as pd

from PIL import Image

def documentation():
    st.write('''
    #### 6PM documentation (사용설명서)
    ''')
    main_img = Image.open('main_image.png')
    st.image(main_img)

    st.write("여기에 순서 같은거 적고 조원들 소개도 쓰면 좋을듯요... UI 디자인 어렵다... ..")

def run_program():
    st.write('''
    #### 1. Search for a picture to upload
    ''')
    upload_img = st.file_uploader(" ")
    if upload_img:
        image_pil = Image.open(upload_img)
        st.image(image_pil)

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

def feedback_page():
    st.write('''
    #### Leave Feedbacks
    ''')
    username_text = st.text_input("User Name")
    st.checkbox("Confirm")
    feedback_text = st.text_area("Feedback Area")
    if not username_text:
        st.warning("You MUST enter your User Name to leave feedbacks")
        st.stop()
    else:
        st.button("Send")



if __name__ == "__main__":
    st.title("6PM - Picture to Music Web Service")
    st.subheader("BDS4010-01 2조 팀프로젝트 최종 과제")

    menu = ["Home", "Feedback"]
    choice = st.sidebar.selectbox("Menu", menu)
    if choice == "Home":
        home_page()
    else:
        feedback_page()






