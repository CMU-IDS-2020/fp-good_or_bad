# CMU Interactive Data Science Final Project

* **Title**: Good or Bad? Visualizing Neural Networks on Sentiment Analysis
* **Online URL**: https://share.streamlit.io/cmu-ids-2020/fp-good_or_bad/main/app.py
* **Team members**:
  * Contact person: hongyuaz@andrew.cmu.edu
  * lingz2@andrew.cmu.edu
  * tianyil2@andrew.cmu.edu
  * yutianzh@andrew.cmu.edu
* **Track**: Narrative (one of Narrative, Model, or Interactive Visualization/Application)
* **Video URL**: https://youtu.be/DUol1CeXiJ4

## Running Instructions

You can 1) check out the online url above (in the case of streamlit error of "exceeding resource limits," you can reboot the app yourself, and don't hesitate to reach out to us if you have questions) or 2) clone this repository, install all required packages and run streamlit run app.py in your terminal.

## Abstract
In this age of social media, gaining an understanding into sentiments can be beneficial and this age has also witnessed a rise of artificial intelligence, which enables a quick capture of the sentiments behind numerous opinions.  However, neural networks are still a “black box” for many people. Current works on visualizing neural networks mainly focus on a certain component and lack user interactions. Given this problem, we aim to build an interactive visualization application, using the task of sentiment analysis as a case study, to help curious machine learning laymen to understand the internal mechanisms of a neural network.

## Work distribution 
The work was split almost equally between four group members. Tianyi worked on model development, visualizations in the training section and dataset visualization. 
Yutian worked on data preprocessing, trained the model with different hyperparameters, saved intermediate results and worked on parts of the narratives on overview and training sections. 
Hongyuan has been responsible for developing visualizations for text preprocessing process, as well as writing narratives in the Overview, Dataset & Input Preprocessing and Predicting sections of the app. 
Ling is in charge of creating visualization for the word embeddings and constructing the web application. Everyone collaborated together on the project design, app development, bug fixes, video presentation and final report. 

## Project process
We started this project by deciding the project topic in sentiment analysis, as our team members are all interested in NLP and especially want to learn more about sentiment analysis and neural networks. Then we finalized our proposal by proposing the idea of “Let people have little or no experience in machine learning learn the whole sentiment analysis process using a neural network”. 
For the project development process, we first collected several datasets as well as implemented and trained a CNN model. Then based on the data we collected in the training process, we further developed visualizations around loss/accuracy plot, model parameter visualization, input preprocessing visualization, word embedding, and prediction probability bar plot. However, we are struggling to find the right track for our project before the design review, as our project involves both models and interactive visualizations. After several meetings among group members and design discussion with course staff in the design review, we decided to choose the narrative track as our project track. 
To adjust and modify our project after the design review, we decided to use a combination of text and visualizations to walk users/viewers through the whole sentiment analysis process. Furthermore, we added the options to enable users/viewers to choose different datasets, adjust the model’s hyperparameters, and compare different sets of parameters during training and prediction. So we total trained around 600 models to give the users full experience in sentiment analysis using neural networks. 
 

## Deliverables

### Proposal

- [x] The URL at the top of this readme needs to point to your application online. It should also list the names of the team members.
- [x] A completed proposal. The contact should submit it as a PDF on Canvas.

### Design review

- [x] Develop a prototype of your project.
- [x] Create a 5 minute video to demonstrate your project and lists any question you have for the course staff. The contact should submit the video on Canvas.

### Final deliverables

- [x] All code for the project should be in the repo.
- [x] A 5 minute video demonstration.
- [ ] Update Readme according to Canvas instructions.
- [ ] A detailed project report. The contact should submit the video and report as a PDF on Canvas.
