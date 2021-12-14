# 6PM-picturetomusic
### BDS4010-01(빅데이터종합설계) 최종 프로젝트
*조원: 나경훈, 이가원, 이동호, 이현지, 정상희, 정재원*
![introduction](https://user-images.githubusercontent.com/80621384/145395976-60b3eb65-940d-4244-a95e-ddcf29f0c9e0.png)


## How-to
driving method explained in Korean

빅데이터종합설계 시간에 발표한 시연 영상입니다.

https://user-images.githubusercontent.com/80621384/146023027-2e13cae5-9e94-4097-bf55-f13e30b328bd.mp4



## Run app
click [this link](https://share.streamlit.io/jjeong-sh/6pm-picturetomusic/main/6PM_app.py)!

6PM_app.py consists of 3 pages
- ```Home``` -> Documentation, **Run Program(our main service)**
- ```Feedback```
- ```Opinions``` (optional)

please read the documents(한국어) one the **[Home]-[Documentation]** task


## About exp1_best_model.pth file (image classification model)
Done transfer-learning with ResNet18 by image datasets from scratch

![image_datasets](https://user-images.githubusercontent.com/80621384/146024381-cb8c7ff2-f675-47db-8ec8-48da93e7811a.png)

collected images from [Flicker](https://www.flickr.com/) which are tagged with below classes, approximately 500 files each

- classes = {**'anxiety', 'depression', 'joy', 'lonely', 'love', 'stress'**}  => 6 classes of sentiment


## Warnings
1. Because it uses Youtube API V.3, qota exceed error can appear. You can get recommendations after 24h if this happens *(googleapiclient.errors.HttpError)*
2. Might encounter other errors. This app is not 100% perfect. Please rerun the app if unhandled exception occurs.
3. DB is NOT built. Usernames are not saved. (it's a prototype for a mini project)
