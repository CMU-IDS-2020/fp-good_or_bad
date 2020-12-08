import streamlit as st
import numpy as np
import pandas as pd
import torch
import copy

from sklearn import decomposition
import plotly.express as px
import plotly.graph_objects as go

import altair as alt
import graphviz
from graphviz import Digraph
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from random import sample
import pickle
from scipy.special import softmax
import time
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 
from torch.nn.functional import pad
import zipfile
import os
from os import listdir
from zipfile import ZipFile
from os.path import isfile, join
from urllib.request import urlopen

from word_highlight import get_highlight_text
from train_vis import get_train_content, loss_acc_plot, params_plot

MODEL_PATH = 'https://github.com/CMU-IDS-2020/fp-good_or_bad/raw/main/models/xentropy_adam_lr0.0001_wd0.0005_bs128'
EMBEDDING_URL = "https://github.com/CMU-IDS-2020/fp-good_or_bad/raw/main/sample_embeddings/sample_words_embeddings.pt"
AMAZON_EMBEDDING_URL = 'https://github.com/CMU-IDS-2020/fp-good_or_bad/blob/main/sample_embeddings/100d/amazon_products_sample_embeddings.pt'
MOVIE_EMBEDDING_URL = 'https://github.com/CMU-IDS-2020/fp-good_or_bad/blob/main/sample_embeddings/100d/movie_review_sample_embeddings.pt'
YELP_EMBEDDING_URL = 'https://github.com/CMU-IDS-2020/fp-good_or_bad/blob/main/sample_embeddings/100d/yelp_restaurant_sample_embeddings.pt'

MODEL_PATH_PT = 'https://github.com/CMU-IDS-2020/fp-good_or_bad/raw/main/models/xentropy_adam_lr0.0001_wd0.0005_bs128.pt'
EPOCH = 30
SAMPLE_LIMIT = 5000
EPOCH_SAMPLE_LIMIT = SAMPLE_LIMIT // EPOCH

MOVIE_DATASET = 'Movie reviews'
AMAZON_DATASET = 'Amazon products'
YELP_DATASET = 'Yelp restaurants'

OVERVIEW = 'Overview'
PREPROCESS = 'Input & Preprocessing'
TRAIN = 'Training'
PREDICT = 'Predicting'

ADAM = 'ADAM'
SGD = 'SGD with Momentum'

preprocesse_exed = False
train_exed = False

class Model:
	def __init__(self, dataset, learning_rate, batch_size, weight_decay, optimizer):
		self.dataset = dataset
		self.learning_rate = learning_rate
		self.batch_size = batch_size
		self.weight_decay = weight_decay
		self.optimizer = optimizer
		self.model_url = None
		self.model_name = None
		self.mapped_dataset = None
		self.mapped_optimizer = None
		self.mapped_weight_decay = None
		self.max_length = 0

		dataset_map = { 'Movie reviews':'movie_reviews','Amazon products' : "amazon_products", 'Yelp restaurants':"yelp_restaurants"}
		optimizer_map = {'ADAM':"adam",'SGD with Momentum':"sgdmomentum"}
		self.mapped_dataset = dataset_map[self.dataset]
		self.mapped_optimizer = optimizer_map[self.optimizer]
		if self.weight_decay == "5e-4":
			self.mapped_weight_decay = "0.0005"
		else:
			self.mapped_weight_decay = self.weight_decay

		url = "https://github.com/CMU-IDS-2020/fp-good_or_bad/raw/main/models/" + self.mapped_dataset + "/model_state_dict/"
		self.model_name = "xentropy_{}_lr{}_wd{}_bs{}.pt".format(self.mapped_optimizer, self.learning_rate, self.mapped_weight_decay, self.batch_size)
		self.model_url = url + self.model_name

		if self.mapped_dataset == 'movie_reviews':
			self.max_len = 29
		elif self.mapped_dataset == "yelp_restaurants":
			self.max_len = 245
		else:
			self.max_len = 721

def main():
	st.sidebar.header('Navigation')
	page = st.sidebar.radio('', (OVERVIEW, PREPROCESS, TRAIN, PREDICT))

	if page == OVERVIEW:
		st.title("Overview")
		st.write("In this age of social media, personal opinions are expressed ubiquitously in the public. \
		Behind these opinions are sentiments and emotions. \
		Gaining an understanding into sentiments regarding a topic can be beneficial in many ways, be it in the case of a business trying to know its customers or the case of a politician trying to know the electorate. \
		This age has also witnessed a rise of artificial intelligence and machine learning, which enables a quick capture of the sentiments behind numerous opinions existing on social media.")
		_, col_center_sent_image, _ = st.beta_columns([1, 2, 1])
		with col_center_sent_image:
			st.image('https://www.kdnuggets.com/images/sentiment-fig-1-689.jpg', caption = 'Sentiment Analysis (reference: https://www.kdnuggets.com/2018/03/5-things-sentiment-analysis-classification.html)', use_column_width=True)
		st.write('''Machine learning methods can be highly accurate and efficient for various tasks. \
		However, machine learning models, especially neural networks, are still a “black box” for many people, even experienced experts in the field (for example, considering the poorly understood nature of generalization of neural networks). \
		Given this problem, we built this visualization application to help people understand internal mechanisms of a neural network. \
		We use the task of sentiment analysis as a case study in our application to walk users through the neural network’s training and decision making process.''')
		_, col_center_nn_image, _ = st.beta_columns([1, 2, 1])
		with col_center_nn_image:
			st.image('https://miro.medium.com/max/726/1*Y4aATgaQ8OO_gxLFTy3rQg.png', caption = 'Neural Networks for Sentiment Analysis (reference: https://medium.com/nlpython/sentiment-analysis-analysis-part-3-neural-networks-3768dd088f71)', use_column_width=True)
		st.write('''To effectively capture, classify and predict sentiments, we design, utilize and demonstrate a convolutional neural network (CNN) [1], which is known for its excellent performance in computer vision tasks, as well as natural language processing tasks recently [1]. \
		Specifically, CNNs have been shown to be able to model inherent syntactic and semantic features of sentimental expressions [2]. \
		Finally, another advantage of using CNNs (and neural networks in general) is no requirement of deep domain knowledge, in this case linguistics [3]. ''')
		st.title("Model Architecture")
		st.write("Our model has the following architecture: ")
		st.write("- 3 layers of 1-Dimensional CNN with kernel sizes (2,3,4) for extracting features")
		st.write("- Max Pooling Layer for retaining prominent features")
		st.write("- Dropout Layer with probability 0.5 for better model generalization")
		st.write("- Linear Layer with output dimension 5 for sentiment classification")
		st.write("Our model uses [Glove Embeddings](https://nlp.stanford.edu/projects/glove/) with 1.9 million vocabulary to obtain pretrained vector representations of words.")
		st.title("Datasets")
		st.write("We trained our model on three relevant datasets, including Rotten Tomato movie reviews, Yelp restaurant reviews and Amazon product reviews, each with various hyperparameter values such as learning rate and weigth decay.")
		st.write("[Rotten Tomato movie reviews](https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews/data) contains more than 15,5000 movie reviews and ratings from 1 to 5.")
		st.write("[Yelp restaurant reviews](https://www.kaggle.com/omkarsabnis/yelp-reviews-dataset) contains more than 11,000 retaurant reviews and ratings from 1 to 5.")
		st.write("[Amazon product reviews](https://www.kaggle.com/datafiniti/consumer-reviews-of-amazon-products?select=Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products.csv) contains more than 5,000 electronic product reviews and ratings from 1 to 5.")
		st.write("**With our app, you will be able to visualize the full process of sentiment analysis using a neural network, as well as the interaction of training data, hyperparameters and the model itself. We hope that this app can demystify the magic of neural networks.\
		Our app is especially useful for curious machine learning laymen.**")
		st.write("**To start using our app, use the sidebar to navigate to a particular section. Then adjust training hyperparameters, and select a specific training dataset. Then write something emotional to check out how a neural net can understand your sentiment!**")

		st.write("")
		st.write("")
		st.write("")
		st.markdown('''
				### References
				[1]
				O'Shea, Keiron, and Ryan Nash. "An introduction to convolutional neural networks." arXiv preprint arXiv:1511.08458 (2015).
				
				[2]
				Weibo Liu, Zidong Wang, Xiaohui Liu, Nianyin Zeng, Yurong Liu and Fuad E. Alsaadi (2017) - "A Survey of Deep Neural Network Architectures and their Applications."
				Neurocomputing, 2017, 234, 11-26.

				[3]
				Hannah Kim and Young-Seob Jeong (2019) - "Sentiment Classification Using Convolutional Neural Networks."
				Applied Sciences, 2019, 9, 2347.
			''')

	elif page != OVERVIEW:
		st.title("Predict Sentiment")
		dataset = st.selectbox('Choose a dataset', ( MOVIE_DATASET, AMAZON_DATASET, YELP_DATASET))
		models = []

		st.sidebar.header("Adjust Model Parameters")
		learning_rate = st.sidebar.select_slider("Learning rate", options=[0.1, 0.01, 0.001, 0.0001], value=0.001)
		# st.sidebar.text('learning rate={}'.format(learning_rate))
		weight_decay = st.sidebar.select_slider("Weight decay", options=[0, 5e-7, 5e-6, 5e-5, 5e-4], value=5e-5)
		# st.sidebar.text('weight decay={}'.format(weight_decay))
		batch_size = st.sidebar.select_slider("Batch_size", options=[32, 64, 128, 256, 512], value=512)
		# st.sidebar.text('batch size={}'.format(batch_size))
		optimizer = st.sidebar.radio('Optimizer', (ADAM, SGD))
		 
		models.append(Model(dataset, learning_rate, batch_size, weight_decay, optimizer))
	 
		two_models = st.sidebar.checkbox('Compare with another set of model parameters')
		if two_models:
			learning_rate2 = st.sidebar.select_slider("Learning rate of second model", options=[0.1, 0.01, 0.001, 0.0001], value=0.001)
			# st.sidebar.text('learning rate={}'.format(learning_rate))
			weight_decay2 = st.sidebar.select_slider("Weight decay of second model", options=[0, 5e-7, 5e-6, 5e-5, 5e-4], value=5e-5)
			# st.sidebar.text('weight decay={}'.format(weight_decay))
			batch_size2 = st.sidebar.select_slider("Batch_size of second model", options=[32, 64, 128, 256, 512], value=512)
			# st.sidebar.text('batch size={}'.format(batch_size))
			optimizer2 = st.sidebar.radio('Optimizer of second model', (ADAM, SGD))
			models.append(Model(dataset, learning_rate2, batch_size2, weight_decay2, optimizer2))

		if dataset == MOVIE_DATASET:
			user_input = st.text_input('Write something emotional and hit enter!', "I absolutely love this romantic movie! It's such an interesting film!")
		elif dataset == AMAZON_DATASET:
			user_input = st.text_input('Write something emotional and hit enter!', "Great device! It's easy to use!")
		else:
			user_input = st.text_input('Write something emotional and hit enter!', "Delicious food! Best place to have lunch with a friend!")




	if page == PREPROCESS:
		preprocessed = run_preprocess(dataset, user_input)
	elif page == TRAIN:
		run_train(models)
	elif page == PREDICT:
		preprocessed = run_preprocess(dataset, user_input, False)
		run_predict(preprocessed, models)

class Network(nn.Module):
	def __init__(self, input_channel, out_channel, kernel_sizes, output_dim):
		super().__init__()
		self.convs = nn.ModuleList([
									nn.Conv1d(in_channels = input_channel, 
											  out_channels = out_channel, 
											  kernel_size = ks)
									for ks in kernel_sizes
									])
		
		self.linear = nn.Linear(len(kernel_sizes) * out_channel, output_dim)
		self.dropout = nn.Dropout(0.5)
		
	def forward(self, embedded):     
		embedded = embedded.permute(0, 2, 1)       
		conved = [F.relu(conv(embedded)) for conv in self.convs]
		pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
		cat = self.dropout(torch.cat(pooled, dim = 1))
		return self.linear(cat)
input_channel = 100
out_channel = 50
kernel_sizes = [2,3,4]
output_dim = 5

def run_preprocess(dataset, input, visible=True):

	# tokenize -> lowercase -> remove stopwords -> lemmatize
	def tokenize_text(text):
		tokenizer = RegexpTokenizer(r'\w+')
		return tokenizer.tokenize(text)


	def lowercase_text(tokens):
		return [token.lower() for token in tokens]


	def remove_stopwords(tokens):
		english_stopwords = stopwords.words('english')
		return [token if token not in english_stopwords and token in word2vec_dict else None for token in tokens]


	def lemmatize(tokens):
		lemmatizer = WordNetLemmatizer()
		return [lemmatizer.lemmatize(token) if token else None for token in tokens]

	if visible:
		st.write("How can neural networks read text like humans? You might wonder. Actually, they cannot; they can only read numbers.\
				 This section walks you through every step that we must perform **before** we convert text to numbers. You will see the actual \
				 converting step in the 'Predicting' section.")

	if visible:
		st.write("Before we head into text preprocessing, let's check out the words that are particularly important, or frequent, in your selected dataset.")
		_, col_center, _ = st.beta_columns([1, 3, 1])
		if dataset == AMAZON_DATASET:
			with col_center:
				st.image('https://github.com/CMU-IDS-2020/fp-good_or_bad/raw/main/static_pictures/amazon_wordcloud.png', use_column_width=True)
			get_highlight_text(input, "top_frequent_words/amazon_products_top1000.pt")
		elif dataset == MOVIE_DATASET:
			with col_center:
				st.image('https://github.com/CMU-IDS-2020/fp-good_or_bad/raw/main/static_pictures/movie_wordcloud.png', use_column_width=True)
			get_highlight_text(input, "top_frequent_words/rotten_tomato_top1000.pt")
		elif dataset == YELP_DATASET:
			with col_center:
				st.image('https://github.com/CMU-IDS-2020/fp-good_or_bad/raw/main/static_pictures/yelp_wordcloud.png', use_column_width=True)
			get_highlight_text(input, "top_frequent_words/yelp_restaurant_top1000.pt")

	if visible:
		st.write('''Let's see all that happens before the step of converting text to numbers, as promised. Now, a very natural question might come to your mind,\
			 "Do you convert on a sentence/word/character level? Would it be too simplified if we convert a whole sentence into a single number?" Indeed, \
			 sentence-level mapping could be meaningless, given that we want to read every word or character in a sentence. Thus, what we usually do in practice \
			 is word or character level mapping. In this app, for the purpose of easy interpretation and demonstration, we choose a word-level mapping for text-to-number conversion.''')

		st.write("Now, the need for breaking sentences into words becomes clear. As you can see in the following figure, our first step is splitting sentences into word tokens by spaces.")
		st.write('''Is that all? Probably not, as the word tokens need some standardization. Consider the tokens "love" and "LOVE." We want them to be considered as the same word, but due to \
			 different letter cases, they are understood as different words by a machine. Thus, the next step follows is making all word tokens have a consistent letter case; we choose to convert all to lowercase.''')
		st.write('''The next step we perform is removing the so-called "stopwords." In English, there are some extremely common yet barely meaningful words, for example, articles. To prevent from diluting, we remove them from our set of word tokens!''')
		st.write('''One last step before text-to-number conversion is lemmatization, which is a further step of standardization. Consider the tokens "cat" and "cats." We want them to be considered as the same word, don't we? Thus, in this last step, we reduce every word token to its stem form.''')
	tokens = tokenize_text(input)
	lowercase_tokens = lowercase_text(tokens)
	removed_stopwords = remove_stopwords(lowercase_tokens)
	lemmatized = lemmatize(removed_stopwords)

	if visible:
		g = Digraph()
		i = 0
		g.node(input)
		for token, lc_token, r_token, l_token in zip(reversed(tokens), reversed(lowercase_tokens), reversed(removed_stopwords), reversed(lemmatized)):
			g.node(token+"token"+str(i), label = token)
			g.edge(input, token+"token"+str(i))
			g.node(lc_token+"lc_token"+str(i), label = lc_token)
			g.edge(token+"token"+str(i), lc_token+"lc_token"+str(i))
			if r_token:
				g.node(r_token+"r_token"+str(i), label = r_token)
				g.edge(lc_token+"lc_token"+str(i), r_token+"r_token"+str(i))
				g.node(l_token+"l_token"+str(i), label = l_token)
				g.edge(r_token+"r_token"+str(i), l_token+"l_token"+str(i))
			i += 1

		with g.subgraph(name='cluster_1') as c:
			c.attr(color='white')
			c.node_attr['style'] = 'filled'
			c.node(input)
			c.attr(label='Original Input')

		with g.subgraph(name='cluster_2') as c:
			c.attr(color='white')
			c.node_attr['style'] = 'filled'
			for i, token in enumerate(reversed(tokens)):
				c.node(token+"token"+str(i))
			c.attr(label='Word Tokens')

		with g.subgraph(name='cluster_3') as c:
			c.attr(color='white')
			c.node_attr['style'] = 'filled'
			for i, token in enumerate(reversed(lowercase_tokens)):
				c.node(token+"lc_token"+str(i))
			c.attr(label='Lowercase Tokens')

		with g.subgraph(name='cluster_4') as c:
			c.attr(color='white')
			c.node_attr['style'] = 'filled'
			for i, token in enumerate(reversed(removed_stopwords)):
				if token:
					c.node(token+"r_token"+str(i))
			c.attr(label='Stopwords Removed')

		with g.subgraph(name='cluster_5') as c:
			c.attr(color='white')
			c.node_attr['style'] = 'filled'
			for i, token in enumerate(reversed(lemmatized)):
				if token:
					c.node(token+"l_token"+str(i))
			c.attr(label='Lemmatized Tokens')

		st.graphviz_chart(g, use_container_width=True)
		st.write('''Now, use the sidebar to navigate to the next section to further explore sentiment analysis via neural nets.''')
	return [token for token in lemmatized if token is not None]

@st.cache(allow_output_mutation=True)
def load_word2vec_dict(word2vec_urls, word2vec_dir):
	word2vec_dict = []
	for i in range(len(word2vec_urls)):
		url = word2vec_urls[i]
		torch.hub.download_url_to_file(url, word2vec_dir+"word2vec_dict"+str(i)+".pt")
		word2vec = pickle.load(open(word2vec_dir+"word2vec_dict"+str(i)+".pt", "rb" ))
		word2vec = list(word2vec.items())
		word2vec_dict += word2vec
	return dict(word2vec_dict)

def tokenize_sentence(sentence, word2vec_dict):
	tokenizer = RegexpTokenizer(r'\w+')
	lemmatizer = WordNetLemmatizer() 
	english_stopwords = stopwords.words('english')
	sentence = sentence.strip()
	tokenized_sentence = [lemmatizer.lemmatize(token.lower()) for token in tokenizer.tokenize(sentence) if token.lower() in word2vec_dict and token.lower() not in english_stopwords]
	return tokenized_sentence

def run_predict(input, models):
	
	def predict(sentence, model_url, max_seq_length = 29):
		#tokenized_sentence = tokenize_sentence(sentence,word2vec_dict)
		embedding_for_plot = {}
		for word in sentence:
			embedding_for_plot[word] = word2vec_dict[word]
		embedding = np.array([word2vec_dict[word] for word in sentence])

		model = Network(input_channel, out_channel, kernel_sizes, output_dim)
		torch.hub.download_url_to_file(model_url, "./cur_model.pt")
		state_dict = torch.load("./cur_model.pt",map_location=torch.device('cpu'))
		model.load_state_dict(state_dict)
		# model.load_state_dict(torch.hub.load_state_dict_from_url(model_url, progress=False, map_location=torch.device('cpu')))
		model.eval()
		
		embedding = np.expand_dims(embedding,axis=0)
		embedding = pad(torch.FloatTensor(embedding), (0, 0, 0, max_seq_length - len(embedding)))
		outputs = model(embedding)
  
		_, predicted = torch.max(outputs.data, 1)
		return softmax(outputs.data), predicted.item() + 1, embedding_for_plot

	st.write("Now let's see what results our neural net gives for your input text. The bar chart below shows the predicted probability that your text contains a certain type of sentiment.\
			 Move your mouse over the bars to see the exact predicted probabilities.")

	st.write("Also try different hyperparameters in the sidebar and see if they predict the same outcome!")

	probs_list = []

	for i in range(len(models)):
		probs, _, embedding = predict(input, models[i].model_url, models[i].max_len)
		probs = probs[0].numpy()
		probs_list.append(probs)

	if len(models) == 2:
		re_columns = st.beta_columns(len(models))
		for i in range(len(models)):
			d = {'Sentiment': ["negative", "somewhat negative", "neutral", "somewhat positive", "positive"], 'Probability': probs_list[i]}
			max_sentiment = d["Sentiment"][np.argmax(d["Probability"])]
			source = pd.DataFrame(d)
			highlight = alt.selection_single(on='mouseover', fields=['Probability'], nearest=False, clear="mouseout")
			c = alt.Chart(source).mark_bar().encode(
				alt.X('Probability:Q', axis=alt.Axis(format='.0%')),
				alt.Y('Sentiment:N', sort=d['Sentiment']),
				color=alt.condition(~highlight,
									alt.Color('Probability:Q', scale=alt.Scale(scheme='greens'), legend=None),
									alt.value('orange'), ), tooltip=['Probability:Q']).properties(width=400, height=200).add_selection(
				highlight).interactive()
			with re_columns[i]:
				st.write(c, use_column_width=True)
				st.write("Our model predicts that your input text contains " + max_sentiment + " sentiment!")
	else:
		_, center_result_col, _ = st.beta_columns([1, 2, 1])
		d = {'Sentiment': ["negative", "somewhat negative", "neutral", "somewhat positive", "positive"], 'Probability': probs_list[0]}
		max_sentiment = d["Sentiment"][np.argmax(d["Probability"])]
		source = pd.DataFrame(d)
		highlight = alt.selection_single(on='mouseover', fields=['Probability'], nearest=False, clear="mouseout")
		c = alt.Chart(source).mark_bar().encode(
			alt.X('Probability:Q', axis=alt.Axis(format='.0%')),
			alt.Y('Sentiment:N', sort=d['Sentiment']),
			color=alt.condition(~highlight, alt.Color('Probability:Q', scale=alt.Scale(scheme='greens'), legend=None),
								alt.value('orange'), ), tooltip=['Probability:Q']).properties(width=650, height=250).add_selection(
			highlight).interactive()
		with center_result_col:
			st.write(c, use_column_width=True)
			st.write("Our model predicts that your input text contains " + max_sentiment + " sentiment!")

	_, center_emb_col, _ = st.beta_columns([1, 3, 1])
	with center_emb_col:
		run_embedding(models[i].mapped_dataset, embedding)

def run_embedding(mapped_dataset, user_input=None):
	@st.cache
	def load_sample_embedding(url):
		embedding_path = "embedding"
		torch.hub.download_url_to_file(url, embedding_path)
		sample_embeddings = pickle.load(open(embedding_path, "rb" ))
		tokens = []
		labels = []
		shapes = []

		for key, val in sample_embeddings.items():
			if key == 'easy':
				st.write(val)
			tokens.append(val)
			labels.append(key)
			shapes.append('0')
		return tokens, labels, shapes

	@st.cache
	def load_usr_embedding(input_dict, sample_tokens, sample_labels, sample_shapes):
		tokens = copy.deepcopy(sample_tokens)
		labels = copy.deepcopy(sample_labels)
		shapes = copy.deepcopy(sample_shapes)
		for key, val in input_dict.items():
			if key == 'easy':
				st.write(val)
			tokens.append(val)
			labels.append(key)
			shapes.append('1')
		return tokens, labels, shapes


	@st.cache
	def transform_3d(tokens):
		# tsne = TSNE(n_components=3, random_state=1, n_iter=100000, metric="cosine")
		pca = decomposition.PCA(n_components=3)
		pca.fit(tokens)
		return pca.transform(tokens)

	@st.cache
	def get_df(values_3d, labels, shapes):
		return pd.DataFrame({
			'x': values_3d[:, 0],
			'y': values_3d[:, 1],
			'z': values_3d[:, 2],
			'label': labels,
			'shapes': shapes
		})
	
	url = 'https://github.com/CMU-IDS-2020/fp-good_or_bad/raw/main/sample_embeddings/100d/{}_sample_embeddings.pt'.format(mapped_dataset)
	sample_tokens, sample_labels, sample_shapes = load_sample_embedding(url)

	if user_input is not None:
		tokens, labels, shapes = load_usr_embedding(user_input, sample_tokens, sample_labels, sample_shapes)
	else:
		tokens = sample_tokens
		labels = sample_labels
		shapes = sample_shapes
	values_3d = transform_3d(tokens)
	source_3d = get_df(values_3d, labels, shapes)

	fig = px.scatter_3d(source_3d, x='x', y='y', z='z',
		color='shapes', symbol='shapes', text='label', labels={'word':'label'},
		width=800, height=600,
		# range_x=[-1500,1500], range_y=[-1500,1500], range_z=[-1500,1500]
		)

	fig.update_traces(marker=dict(size=2), selector=dict(mode='markers'))
	# fig.update_traces(hovertemplate=' ')
	fig.update_traces(hoverinfo='skip', hovertemplate=None, selector=dict(type='scatter3d'))
	fig.update_layout(scene_aspectmode='cube', showlegend=False)
	# fig.update_scenes(xaxis_visible=False, yaxis_visible=False,zaxis_visible=False )

	st.plotly_chart(fig, use_column_width=True)

def run_train(models):
	# dataset_path = "amazon_products" or "movie_reviews" or "yelp_restaurants"
	# optimizer_path = "xentropy_adam_all" or "xentropy_sgdmomentum_all"

	st.header("Let's explore the training process of our CNN network with chosen hyper-parameters!")
	st.subheader("Accuracy & Loss")
	st.write("The loss (objective) function we used for our model is cross entropy loss. Here we plot the loss for training and validation sets, which reflect how **well** the model is doing in these two sets. Since we always want to minimize the loss, a good training process usually has decreasing loss values over steps. \
	The accuracy metric here indicates the percentage of correct predictions, and measures how accurate the model’s predictions are compared to true labels.")
	st.write("_**Tips**_")
	st.write("1. Hover your mouse on the plot to compare the value of accuracy/loss and train/validation over epochs.")
	st.write("2. If you notice an increase in validation loss, a decrease in validation accuracy or oscillation of loss and accuracy, it’s a bad sign and usually indicates the model is overfitting. **Try to change and tune hyperparameters**!")
	param_dfs = []

	for model in models:
		opt_path = "xentropy_{}_all".format(model.mapped_optimizer)
		CONTENT = get_train_content(dataset_path=model.mapped_dataset, optimizer_path=opt_path)
		param_dfs.append(CONTENT[model.model_name[:-3]])

	# get number of models
	if len(models) == 1: 
		_, center_col, _ = st.beta_columns([1, 3, 1])
		with center_col:
			st.write(loss_acc_plot(param_dfs[0], False))
	elif len(models) == 2:
		col1, col2 = st.beta_columns(2)
		with col1:
			st.write(loss_acc_plot(param_dfs[0]))
		with col2:
			st.write(loss_acc_plot(param_dfs[1]))

	# add description here 
	st.subheader("Model Paramaters")
	st.write("The model parameters are usually weights and bias . Our model consists of 4 layers (3 convolutional layers and 1 linear layer), so we visualize the distribution of weights and bias for these 4 layers here using heat maps and marginal histograms. \
	For a good training process, we should see the model parameters are clustered around zero at the first epoch and are become more **dispersed** over steps, indicating that they are learning different values to coverge to the optimal point! ")
	st.write("_**Tips**_")
	st.write("1. Hover over the plot to see the distribution of model parameters in marginal histograms.")
	st.write("2. If you notice that the distributions of the model parameters are not changing over steps, it’s a bad sign! This usually indicates that the model isn’t learning :( Try different hyperparameters!")

	title = ["**_First Layer_: Convolution layer with kernel size 2**", "**_Second Layer_: Convolution layer with kernel size 3**",
			 "**_Third Layer_: Convolution layer with kernel size 4**", "**_Fourth Layer_: Fully connected layer**"]
	if len(models) == 1:
		middle = params_plot(param_dfs[0], False)
		for i in range(len(middle)):
			st.write(title[i])
			p = middle[i]
			_, center_col, _ = st.beta_columns([1, 3, 1])
			with center_col:
				st.write(p)
	elif len(models) == 2:
		left = params_plot(param_dfs[0])
		right = params_plot(param_dfs[1])
		for i in range(len(left)):
			st.write(title[i])
			col1, col2 = st.beta_columns(2)
			with col1:
				st.write(left[i])
			with col2:
				st.write(right[i])


if __name__ == "__main__":
	st.set_page_config(layout="wide")
	word2vec_dict = load_word2vec_dict(word2vec_urls = ['https://github.com/CMU-IDS-2020/fp-good_or_bad/raw/main/word2vec/100d/word2vec_100d_{}.pt'.format(i+1) for i in range(5)], word2vec_dir = "./word2vec")
	main()