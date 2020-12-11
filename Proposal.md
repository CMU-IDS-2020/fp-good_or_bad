# Final Project Proposal

**Project URL**: https://share.streamlit.io/cmu-ids-2020/fp-good_or_bad/main/app.py

##Background
In this age of social media, personal opinions are expressed ubiquitously in the public. Behind these opinions are sentiments and emotions. Gaining an understanding into sentiments regarding a topic can be beneficial in many ways, be it in the case of a business trying to know its customers or the case of a politician trying to know the electorate. This age has also witnessed a rise of artificial intelligence and machine learning, which enables a quick capture of the sentiments behind numerous opinions existing on social media.

##Problem
Machine learning methods can be highly accurate and efficient for various tasks. However, machine learning models, especially neural networks, are still a “black box” for many people, even experienced experts in the field (for example, considering the poorly understood nature of generalization of neural networks). Given this problem, we aim to build a visualization application to help people understand internal mechanisms of a neural network. We use the task of sentiment analysis as a case study in our application to walk users through the neural network’s training and decision making process.

##Solution
We hope to train a deep learning model offline that could classify the sentiments on different types of comments. Since the target of our project is to visualize and provide interactions on the training and predicting process of deep learning models, we’ll use various state-of-the-art model architectures to train our model and record their intermediate results for visualization. We’ll allow the users to explore various methods and parameters used in different phrases of model training and compare the model’s prediction output. 

##Model & Dataset
We plan to use word embeddings combined with CNN, which is the state-of-the-art model in text classification. The training steps we will visualize include tokenization, word embedding and gradient descent. The model parameters the users may explore include different optimization functions (sgd, adam, momentum), loss functions and hyperparameters such as batch size, learning rate and weight decay. We’ll implement our model using pytorch and train it offline on google colab or aws. We’ll run our model on three datasets:
1. Rotten tomato movie reviews come from Kaggle, which includes more than 65,500 reviews and ratings from 0 to 4. 
2. Yelp restaurant reviews from Yelp open dataset, which includes 8,021,122 reviews from star ratings 1 to 5. 
3. Amazon electronics product review dataset from UCSD, which includes 6,739,590 reviews from ratings 1 to 5. 

##Timeline  
Each member of our team will be responsible for a section of the model training process mentioned above, and we’ll implement the visualization together. We plan to gather all the datasets and finish data preprocessing before Nov 11st and then train and deploy a prototype model before Nov 19th.

