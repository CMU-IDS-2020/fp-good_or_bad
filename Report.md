# Final Project Report

Team Good-or-Bad

**Project URL**: [https://share.streamlit.io/cmu-ids-2020/fp-good_or_bad/main/app.py](https://share.streamlit.io/cmu-ids-2020/fp-good_or_bad/main/app.py)

In this age of social media, gaining an understanding into sentiments can be beneficial and this age has also witnessed a rise of artificial intelligence, which enables a quick capture of the sentiments behind numerous opinions.  However, neural networks are still a “black box” for many people. Current works on visualizing neural networks mainly focus on a certain component and lack user interactions. Given this problem, we aim to build an interactive visualization application, using the task of sentiment analysis as a case study, to help curious machine learning laymen to understand the internal mechanisms of a neural network.

## Introduction

In this age of social media, personal opinions are expressed ubiquitously in the public. Behind these opinions are sentiments and emotions. Gaining an understanding into sentiments regarding a topic can be beneficial in many ways, be it in the case of a business trying to know its customers or the case of a politician trying to know the electorate. This age has also witnessed a rise of artificial intelligence and machine learning, which enables a quick capture of the sentiments behind numerous opinions existing on social media.

Machine learning methods can be highly accurate and efficient for various tasks. However, machine learning models, especially neural networks, are still a “black box” for many people, even experienced experts in the field (for example, considering the poorly understood nature of generalization of neural networks). Many researchers and scholars have published enormous amounts of papers and visualization applications to gain an understanding of neural networks. However, current works mainly focus on visualizing a certain component of neural networks, such as gradient descent or pattern detection, instead of the end-to-end process of training neural networks. Moreover, existing visualizations lack user interactions.

 
Given this problem, we aim to build a visualization application for curious machine learning laymen to understand the internal mechanisms of a neural network. We use the task of sentiment analysis as a case study in our application to walk users through the full process of neural network’s training and decision making process, as well as the interaction of training data, hyperparameters and the model itself. We hope that this app can demystify the magic of neural networks for users with little background knowledge of machine learning.

## Related Work

Two types of work are related: visualizing neural networks, especially convolutional neural networks, and sentiment analysis.

Neural networks have been considered a black-box algorithm and a rich literature works on visualizing and understanding the internal mechanism of neural networks. Most of the works focus on a specific component of neural networks, such as loss [1,3] , gate activations and saliency [2,3,4], model features [5] and optimization functions [3,6,7,8,9]. Moreover, convolutional neural network visualizations are mostly image processing [10, 11,12], we have not yet found one visualizing its application in language processing.

Sentiment analysis has been a popular topic nowadays. Lots of papers and visualization applications provide descriptions on its process [13,14,15,16,17]. However, they mainly contain static visualizations, such as heatmaps and barcharts. Plots are generated with the best model parameters and hyperparameters selected by the authors in advance. Descriptions and visualizations on sentiment classification remain on a theoretical level, few of them provide concrete examples and allow users to interact with the training data, hyperparameters and model itself.

## Methods

We display the input preprocessing, training and predicting processes of one specific neural network architecture to help users understand sentiment analysis with machine learning. 

### Model and Dataset

We referred to this [tutorial](https://github.com/bentrevett/pytorch-sentiment-analysis/blob/master/4%20-%20Convolutional%20Sentiment%20Analysis.ipynb) for the neural network model displayed in our application. Our model has the following architecture:

-   3 layers of 1-Dimensional CNN with kernel sizes (2,3,4) for extracting features
-   Max Pooling Layer for retaining prominent features
-   Dropout Layer with probability 0.5 for better model generalization
-   Linear Layer with output dimension 5 for sentiment classification
 
Our model uses GloVe word embeddings [18] with 1.9 million vocabulary to obtain pre-trained vector representations of words. In particular, semantically similar words are closer to each other in the GloVe embedding space.

We trained our model on three relevant datasets, including [Rotten Tomato movie reviews](https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews/data), [Yelp restaurant reviews](https://www.kaggle.com/omkarsabnis/yelp-reviews-dataset) and [Amazon product reviews](https://www.kaggle.com/datafiniti/consumer-reviews-of-amazon-products). Rotten Tomato movie reviews dataset contains more than 15,5000 movie reviews and ratings from 1 to 5. Yelp restaurant reviews dataset contains more than 11,000 restaurant reviews and ratings from 1 to 5. Amazon product reviews dataset contains more than 5,000 electronic product reviews and ratings from 1 to 5.

### Visualization Methods

We used the combination of texts and interactive visualizations to walk users through the whole sentiment analysis process. We split this process into four steps: Overview, Dataset & Input Preprocessing, Training, and Prediction.

#### The Overview Section
This section aims to give users the context and background around sentiment analysis and neural networks. We also include some instructions to guide the users through their exploration of this topic.

#### The Dataset & Input Preprocessing  Section

This section is further separated into three parts: dataset visualization, input preprocessing, and word embedding. For dataset visualization, we build word cloud and word importance visualization based on word frequency. Both graphs help the users to better understand the relationship between input text and chosen dataset. For input preprocessing, we use a flowchart to visualize the whole processing steps before converting text to word embedding. This process includes tokenization, stopword removals, and lemmatization. Finally, we incorporate a word embedding plot in this section to further explain how neural networks can understand and "read" words. We use word2vec, developed by Google, to translate each word into a vector of its position in the embedding space. We perform dimensionality reduction tricks to map the word embeddings to a 3D space while keeping their relative positions. This plot aims to provide users knowledge about the relationship between the semantic distance of words and their actual distance in the embedding space.

#### The Training  Section 

This section illustrates the whole training process of neural networks. First, we use line plots to show how loss and accuracy change over steps for both training and validation datasets. This visualization provides users information about how well the model is learning and demonstrates the model's convergence and generalization ability. Next, we use heatmaps and marginal histograms to show the distribution of the model's parameters. By showing how model parameters change over time, this visualization reveals the learning process of neural networks to the users and helps them identify potential exploding or vanishing gradient problems. Starting from this section, the users are able to select different learning rates, weight decay, and optimizer functions as well as compare different sets of hyperparameters, which enable them to explore how those hyper-parameters affect the learning process of neural networks.

#### The Predicting Section

This section aims to provide the final prediction results from a trained neural network. We use an interactive color encoded bar plot to show the predicted probability (softmax value) for each sentiment. Same as the training section, this section also enables users abilities to change model hyperparameters and compare the results.


## Results and Discussion

We ran a user survey to evaluate effectiveness of our developed application. We recruited 19 participants who have no or little experience in machine learning and sentiment analysis, each of whom used our app once and completed a survey. In the survey, we asked the participants to rate their level of understanding of neural networks and sentiment analysis before and after using our app. These questions are related to the ultimate goal of our app. We observed statistically significant improvement in both, with paired t-test p-value = 8.084e-09 for understanding of neural networks and p-value = 5.716e-10 for understanding of sentiment analysis.

In the survey, we also ask questions that target various detailed aspects of our app. Another question asks the participant to rate to what degree our app helps them understand how neural networks operate for this task of sentiment analysis. 57.9% of the participants answered “extremely agree,” and 31.6% of the participants answered “somewhat agree.” In addition, we ask the participants to rate to what degree our app helps them understand how hyperparameters affect performance of neural networks. For this question, 52.6% of the participants answered “extremely agree,” and 36.8% of the participants answered “somewhat agree.” For evaluating visualization design, we also ask to what degree our participants agree that our visualizations help to clarify the concepts, and 73.7% of the participants extremely agreed. Moreover, we ask the participants to rate to what degree they agree that our app provides clear navigation, with 47.4% of the participants extremely agreed and 36.8% of the participants somewhat agreed.

It’s worth noting that we ask the participants to rate the visualization components in terms of easiness of understanding. The model weights/bias and word embeddings were least rated, with 30+% of the participants rating them as straightforward to understand. At the time of conducting the survey, we did not have as many text explanations as we have now; from this survey result, we quickly realized this problem and added more explanations about relevant concepts. Finally, we also noticed that our participants especially like the function that they can explore training and predicting neural networks with different input text.

According to these results, we believe that we have achieved our goal for the app, that is, helping users better understand neural networks on the sentiment analysis task.

## Future Work

Our Good-or-Bad application presents users with a clear illustration of the internal mechanism of sentiment analysis using neural networks. It provides a good variety of model parameters to produce both pragmatic and representative models while keeping the resource usage of the application under control. It also clearly visualized the whole sentiment analysis workflow from preprocessing to predicting, so that users could relate what they have learned about neural networks on a real-life example.

The goal of our application is to present educational visualizations of neural networks on sentiment analysis. Yet there may be patterns in the visualizations themselves that can provide useful insights to the users. The application can be extended to add the functionality that points out any interesting patterns in the visualization when visualization is created using the user inputs. But it is a challenge to balance the amount of guidance to users and the flexibility of interpretation of the visualizations.

Our current implementation of the application has a fixed model architecture with three convolutional layers, one max pooling layer, one dropout layer, and one output layer. We expect the fixed model architecture function as control factors so that our target users could compare the impacts of different model parameters. While this is sufficient for understanding model parameters, we can also extend our application with a subsection that compares the effect of adding or removing different types of layers on the model and the prediction. This can be easily achieved if we allow a reasonable number of layer combinations, as all models in our application are pre-trained and loaded on demand.

In this project, we have limited the number of possible models users could observe due to resource constraints. The application is enough for machine learning amateurs to get a high-level understanding of the neural network prediction process, but it would be ideal to allow more fine-grained tuning of the models so that users could isolate more parameters while experimenting with the application. The challenge lies in the time and computational resources required. We could achieve this with reasonable application performance by adding more pre-trained data with different model parameters while limiting the flexibility of model tuning; we could also grant flexibility to users and allow arbitrary model parameters and model architectures, then train the models on the spot which takes a long time. This is a trade-off that we should always consider when implementing similar visualization applications.

## References

[1] Li, Hao, et al. "Visualizing the loss landscape of neural nets." Advances in neural information processing systems 31 (2018): 6389-6399.

  

[2] Li, Jiwei, et al. "Visualizing and understanding neural models in nlp." arXiv preprint arXiv:1506.01066 (2015)

  

[3] Chatzimichailidis, Avraam, et al. "GradVis: Visualization and Second Order Analysis of Optimization Surfaces during the Training of Deep Neural Networks." 2019 IEEE/ACM Workshop on Machine Learning in High Performance Computing Environments (MLHPC). IEEE, 2019.

  

[4] Karpathy, Andrej, Justin Johnson, and Li Fei-Fei. "Visualizing and understanding recurrent networks." arXiv preprint arXiv:1506.02078 (2015).

  

[5] Zeiler, Matthew D., and Rob Fergus. "Visualizing and understanding convolutional networks." European conference on computer vision. Springer, Cham, 2014.

  

[6] “VISO: visualize optimization algorithms for objective functions”. [http://www.yalibian.com/research/viso/](http://www.yalibian.com/research/viso/). Accessed 10 Dec, 2020.

  

[7] Logan Yang. “Visualizing Optimization Trajectory of Neural Nets”. TowardsDataScience, Friom Animation to Intuition, [https://towardsdatascience.com/from-animation-to-intuition-visualizing-optimization-trajectory-in-neural-nets-726e43a08d85](https://towardsdatascience.com/from-animation-to-intuition-visualizing-optimization-trajectory-in-neural-nets-726e43a08d85). Accessed 10 Dec, 2020.

  

[8] An Interactive Tutorial on Numerical Optimization: http://www.benfrederickson.com/numerical-optimization/

[9] Ruder, Sebastian. "An overview of gradient descent optimization algorithms." arXiv preprint arXiv:1609.04747 (2016).

[10] Wang, Zijie J., et al. "CNN Explainer: Learning Convolutional Neural Networks with Interactive Visualization." arXiv preprint arXiv:2004.15004 (2020).

[11] Zhang, Xufan, et al. "NeuralVis: visualizing and interpreting deep learning models." 2019 34th IEEE/ACM International Conference on Automated Software Engineering (ASE). IEEE, 2019.

[12] Gigante, Scott, et al. "Visualizing the PHATE of Neural Networks." Advances in Neural Information Processing Systems. 2019.

[13] Badjatiya, Pinkesh, et al. "Deep learning for hate speech detection in tweets." Proceedings of the 26th International Conference on World Wide Web Companion. 2017.

[14] Cachola, Isabel, et al. "Expressively vulgar: The socio-dynamics of vulgarity and its effects on sentiment analysis in social media." Proceedings of the 27th International Conference on Computational Linguistics. 2018.

[15] Mozetič, Igor, Miha Grčar, and Jasmina Smailović. "Multilingual Twitter sentiment classification: The role of human annotators." PloS one 11.5 (2016): e0155036.

[16] Poria, Soujanya, et al. "Meld: A multimodal multi-party dataset for emotion recognition in conversations." arXiv preprint arXiv:1810.02508 (2018).

[17] Araque, Oscar, et al. "Depechemood++: a bilingual emotion lexicon built through simple yet powerful techniques." IEEE transactions on affective computing (2019).

[18] Pennington, Jeffrey, et al. 2014. GloVe: Global Vectors for Word Representation, EMNLP: 1532-1543 (2014).

