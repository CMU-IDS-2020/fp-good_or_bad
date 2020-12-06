import streamlit as st
import numpy as np
import pandas as pd
import torch
import copy

from sklearn.manifold import TSNE
import plotly.express as px
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
EPOCH = 50
SAMPLE_LIMIT = 5000
EPOCH_SAMPLE_LIMIT = SAMPLE_LIMIT // EPOCH

AMAZON_DATASET = 'Amazon products'
MOVIE_DATASET = 'Movie reviews'
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

		dataset_map = {'Amazon products' : "amazon_products", 'Movie reviews':'movie_reviews', 'Yelp restaurants':"yelp_restaurants"}
		optimizer_map = {'ADAM':"adam",'SGD with Momentum':"sgdmomentm"}
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
		st.image('https://www.kdnuggets.com/images/sentiment-fig-1-689.jpg', caption = 'Sentiment Analysis', use_column_width=True)
		st.write('''Machine learning methods can be highly accurate and efficient for various tasks. \
		However, machine learning models, especially neural networks, are still a “black box” for many people, even experienced experts in the field (for example, considering the poorly understood nature of generalization of neural networks). \
		Given this problem, we built this visualization application to help people understand internal mechanisms of a neural network. \
		We use the task of sentiment analysis as a case study in our application to walk users through the neural network’s training and decision making process.''')
		st.image('https://miro.medium.com/max/726/1*Y4aATgaQ8OO_gxLFTy3rQg.png', caption = 'Neural Networks for Sentiment Analysis', use_column_width = True)
		st.write('''To effectively capture, classify and predict sentiments, we design, utilize and demonstrate a convolutional neural network (CNN), which is known for its excellent performance in computer vision tasks, as well as natural language processing tasks recently [1]. \
		Specifically, CNNs have been shown to be able to model inherent syntactic and semantic features of sentimental expressions [2]. \
		Finally, another advantage of using CNNs (and neural networks in general) is no requirement of deep domain knowledge, in this case linguistics [2]. ''')
		st.write("Our model has the following architecture: ")
		st.write("- 1-Dimensional CNN Layer for extracting features")
		st.write("- Max Pooling Layer for retaining prominent features")
		st.write("- Dropout Layer for better model generalization")
		st.write("- Linear Layer for sentiment classification")
		st.write("We trained our model on three relevant datasets, including Rotten Tomato movie reviews, Yelp restaurant reviews and Amazon product reviews, each with various hyperparameter values. \
		With our app, you will be able to visualize the full process of sentiment analysis using a neural network, as well as the interaction of training data, hyperparameters and the model itself. \
		We hope that this app can demystify the magic of neural networks.")
		st.write("To start using our app, first select a specific training dataset, then adjust hyparameter values using the sidebar!")
	elif page != OVERVIEW:
		st.title("Predict Sentiment")
		dataset = st.selectbox('Choose a dataset', (AMAZON_DATASET, MOVIE_DATASET, YELP_DATASET))
		models = []

		st.sidebar.header("Adjust Model Parameters")
		learning_rate = st.sidebar.select_slider("Learning rate", options=[1, 0.1, 0.01, 0.001, 0.0001], value=0.001)
		# st.sidebar.text('learning rate={}'.format(learning_rate))
		weight_decay = st.sidebar.select_slider("Weight decay", options=[5e-7, 5e-6, 5e-5, 5e-4], value=5e-5)
		# st.sidebar.text('weight decay={}'.format(weight_decay))
		batch_size = st.sidebar.select_slider("Batch_size", options=[32, 64, 128, 256, 512], value=512)
		# st.sidebar.text('batch size={}'.format(batch_size))
		optimizer = st.sidebar.radio('Optimizer', (ADAM, SGD))
		 
		models.append(Model(dataset, learning_rate, batch_size, weight_decay, optimizer))
	 
		two_models = st.sidebar.checkbox('Compare with another set of model parameters')
		if two_models:
			learning_rate2 = st.sidebar.select_slider("Learning rate of second model", options=[1, 0.1, 0.01, 0.001, 0.0001], value=0.001)
			# st.sidebar.text('learning rate={}'.format(learning_rate))
			weight_decay2 = st.sidebar.select_slider("Weight decay of second model", options=[5e-7, 5e-6, 5e-5, 5e-4], value=5e-5)
			# st.sidebar.text('weight decay={}'.format(weight_decay))
			batch_size2 = st.sidebar.select_slider("Batch_size of second model", options=[32, 64, 128, 256, 512], value=512)
			# st.sidebar.text('batch size={}'.format(batch_size))
			optimizer2 = st.sidebar.radio('Optimizer of second model', (ADAM, SGD))
			models.append(Model(dataset, learning_rate2, batch_size2, weight_decay2, optimizer2))

		
		user_input = st.text_input('Write something emotional and hit enter!', 'I absolutely love it!')


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
input_channel = 300
out_channel = 100
kernel_sizes = [3,4,5]
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
		if dataset == AMAZON_DATASET:
			st.image('https://github.com/CMU-IDS-2020/fp-good_or_bad/raw/main/static_pictures/amazon_wordcloud.png')
			get_highlight_text(input, "top_frequent_words/amazon_products_top1000.pt")
		elif dataset == MOVIE_DATASET:
			st.image('https://github.com/CMU-IDS-2020/fp-good_or_bad/raw/main/static_pictures/movie_wordcloud.png')
			get_highlight_text(input, "top_frequent_words/rotten_tomato_top1000.pt")
		elif dataset == YELP_DATASET:
			st.image('https://github.com/CMU-IDS-2020/fp-good_or_bad/raw/main/static_pictures/yelp_wordcloud.png')
			get_highlight_text(input, "top_frequent_words/yelp_restaurant_top1000.pt")		

	tokens = tokenize_text(input)
	lowercase_tokens = lowercase_text(tokens)
	removed_stopwords = remove_stopwords(lowercase_tokens)
	lemmatized = lemmatize(removed_stopwords)

	if visible:
		g = Digraph()
		i = 0
		g.node(input)
		for token, lc_token, r_token, l_token in zip(tokens, lowercase_tokens, removed_stopwords, lemmatized):
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
			for i, token in enumerate(tokens):
				c.node(token+"token"+str(i))
			c.attr(label='Word Tokens')

		with g.subgraph(name='cluster_3') as c:
			c.attr(color='white')
			c.node_attr['style'] = 'filled'
			for i, token in enumerate(lowercase_tokens):
				c.node(token+"lc_token"+str(i))
			c.attr(label='Lowercase Tokens')

		with g.subgraph(name='cluster_4') as c:
			c.attr(color='white')
			c.node_attr['style'] = 'filled'
			for i, token in enumerate(removed_stopwords):
				if token:
					c.node(token+"r_token"+str(i))
			c.attr(label='Stopwords Removed')

		with g.subgraph(name='cluster_5') as c:
			c.attr(color='white')
			c.node_attr['style'] = 'filled'
			for i, token in enumerate(lemmatized):
				if token:
					c.node(token+"l_token"+str(i))
			c.attr(label='Lemmatized Tokens')

		st.graphviz_chart(g)
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
	
	def predict(sentence, model_url = 'https://github.com/CMU-IDS-2020/fp-good_or_bad/raw/main/models/xentropy_adam_lr0.0001_wd0.0005_bs128.pt', max_seq_length = 29):
		#tokenized_sentence = tokenize_sentence(sentence,word2vec_dict)
		embedding_for_plot = {}
		for word in sentence:
			embedding_for_plot[word] = word2vec_dict[word]
		embedding = np.array([word2vec_dict[word] for word in sentence])

		model = Network(input_channel, out_channel, kernel_sizes, output_dim)
		model.load_state_dict(torch.hub.load_state_dict_from_url(model_url, progress=False, map_location=torch.device('cpu')))
		model.eval()
		
		embedding = np.expand_dims(embedding,axis=0)
		embedding = pad(torch.FloatTensor(embedding), (0, 0, 0, max_seq_length - len(embedding)))
		outputs = model(embedding)
  
		_, predicted = torch.max(outputs.data, 1)
		return softmax(outputs.data), predicted.item() + 1, embedding_for_plot

	probs_list = []
	emb_columns = st.beta_columns(len(models))
	for i in range(len(models)):
		probs, _, embedding = predict(input, models[i].model_url)
		with emb_columns[i]:
			run_embedding(model.mapped_dataset, embedding)

		probs = probs[0].numpy()
		probs_list.append(probs)


	re_columns = st.beta_columns(len(models))
	for i in range(len(models)):

		d = {'Sentiment': ["negative", "somewhat negative", "neutral", "somewhat positive", "positive"], 'Probability': probs}
		max_sentiment = d["Sentiment"][np.argmax(d["Probability"])]
		source = pd.DataFrame(d)
		c = alt.Chart(source).mark_bar().encode(
			alt.X('Probability:Q', axis=alt.Axis(format='.0%')),
			alt.Y('Sentiment:N', sort = d['Sentiment']),
			color=alt.condition(
				alt.datum.Sentiment == max_sentiment,  # If the year is 1810 this test returns True,
				alt.value('orange'),     # which sets the bar orange.
				alt.value('steelblue')   # And if it's not true it sets the bar steelblue.
			)
		)
		with re_columns[i]:
			st.write(c)
			st.write("Our model predicts that your input text contains " + max_sentiment + " sentiment!")

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
			tokens.append(val)
			labels.append(key)
			shapes.append(0)
		return tokens, labels, shapes

	@st.cache
	def load_usr_embedding(input_dict, sample_tokens, sample_labels, sample_shapes):
		tokens = copy.deepcopy(sample_tokens)
		labels = copy.deepcopy(sample_labels)
		shapes = copy.deepcopy(sample_shapes)
		for key, val in input_dict.items():
			tokens.append(val)
			labels.append(key)
			shapes.append(1)
		return tokens, labels, shapes


	@st.cache
	def transform_3d(tokens):
		tsne = TSNE(n_components=3, random_state=1, n_iter=100000, metric="cosine")
		return tsne.fit_transform(tokens)

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
	sample_tokens, sample_labels, sample_shapes = load_sample_embedding(EMBEDDING_URL)

	if user_input is not None:
		tokens, labels, shapes = load_usr_embedding(user_input, sample_tokens, sample_labels, sample_shapes)
	else:
		tokens = sample_tokens
		labels = sample_labels
		shapes = sample_shapes
	values_3d = transform_3d(tokens)
	source_3d = get_df(values_3d, labels, shapes)

	fig = px.scatter_3d(source_3d, x='x', y='y', z='z',
		color='label', symbol='shapes', text='label', labels={'word':'label'},
		width=1000, height=800, 
		range_x=[-1500,1500], range_y=[-1500,1500], range_z=[-1500,1500])

	fig.update_traces(marker=dict(size=4), selector=dict(mode='markers'))
	# fig.update_traces(hovertemplate=' ')
	fig.update_traces(hoverinfo='skip', hovertemplate=None, selector=dict(type='scatter3d'))
	fig.update_layout(scene_aspectmode='cube')
	st.plotly_chart(fig)

def run_train(models):
	# dataset_path = "amazon_products" or "movie_reviews" or "yelp_restaurants"
	# optimizer_path = "xentropy_adam_all" or "xentropy_sgdmomentum_all"

	param_dfs = []

	for model in models:
		opt_path = "xentropy_{}_all".format(model.mapped_optimizer)
		CONTENT = get_train_content(dataset_path=model.mapped_dataset, optimizer_path=opt_path)
		param_dfs.append(CONTENT[model.model_name[:-3]])

	# get number of models
	if len(models) == 1: 
		st.write(loss_acc_plot(param_dfs[0]))
	elif len(models) == 2:
		col1, col2 = st.beta_columns(2)
		with col1:
			st.write(loss_acc_plot(param_dfs[0]))
		with col2:
			st.write(loss_acc_plot(param_dfs[1]))

	# add description here 

	if len(models) == 1: 
		st.write(params_plot(param_dfs[0]))
	elif len(models) == 2:
		col1, col2 = st.beta_columns(2)
		with col1:
			st.write(params_plot(param_dfs[0]))
		with col2:
			st.write(params_plot(param_dfs[1]))

# def run_train():
# 	@st.cache(allow_output_mutation=True)
# 	def get_content():
# 		return torch.hub.load_state_dict_from_url(MODEL_PATH, progress=False, map_location=torch.device('cpu'))

# 	def get_param_df(content):
# 		model_parameters = content['model_parameters']

# 		param_df = pd.DataFrame({'epoch': [], 'param_type': [], 'value': []},
# 						  columns=['epoch', 'param_type', 'value'])

# 		for i in range(len(model_parameters)):
# 			epoch = i+1
# 			params = model_parameters[i]
# 			for key in params.keys():
# 				param_type = key
# 				values = params[key].numpy().reshape(-1).tolist()
# 				if len(values) > EPOCH_SAMPLE_LIMIT:
# 					values = sample(values, EPOCH_SAMPLE_LIMIT)
# 				rows = pd.DataFrame({'epoch': [epoch]*len(values), 'param_type': [param_type]*len(values), 'value': values})
# 				param_df = param_df.append(rows, ignore_index=True)

# 		# convs.0.weight
# 		# convs.0.bias
# 		# convs.1.weight
# 		# convs.1.bias
# 		# convs.2.weight
# 		# convs.2.bias
# 		# linear.weight
# 		# linear.bias
# 		convs_0_weight_df = param_df[param_df['param_type'] == 'convs.0.weight']
# 		convs_0_bias_df = param_df[param_df['param_type'] == 'convs.0.bias']
# 		convs_1_weight_df = param_df[param_df['param_type'] == 'convs.1.weight']
# 		convs_1_bias_df = param_df[param_df['param_type'] == 'convs.1.bias']
# 		convs_2_weight_df = param_df[param_df['param_type'] == 'convs.2.weight']
# 		convs_2_bias_df = param_df[param_df['param_type'] == 'convs.2.bias']
# 		linear_weight_df = param_df[param_df['param_type'] == 'linear.weight']
# 		linear_bias_df = param_df[param_df['param_type'] == 'linear.bias']

# 		param_df_list = [convs_0_weight_df, convs_0_bias_df, convs_1_weight_df, convs_1_bias_df, convs_2_weight_df, convs_2_bias_df, linear_weight_df, linear_bias_df]
# 		param_df_name = ["convs.0.weight", "convs.0.bias", "convs.1.weight", "convs.1.bias", "convs.2.weight", "convs.2.bias", "linear.weight", "linear.bias"]
# 		return param_df_list, param_df_name


# 	def get_loss_acc_df(content):
# 		train_loss = content['train_loss']
# 		train_acc = content['train_acc']
# 		validation_loss = content['test_loss']
# 		validation_acc = content['test_acc']
# 		avg_train_time = content['ave_train_time']

# 		df = pd.DataFrame({'train_loss': train_loss, 'train_acc': train_acc, 'validation_loss': validation_loss,
# 						   'validation_acc': validation_acc, 'epoch': range(1, EPOCH + 1)},
# 						  columns=['train_loss', 'train_acc', 'validation_loss', 'validation_acc', 'epoch'])

# 		df_loss = df.melt(id_vars=["epoch"],
# 						  value_vars=["train_loss", "validation_loss"],
# 						  var_name="type",
# 						  value_name="loss")
# 		df_acc = df.melt(id_vars=["epoch"],
# 						 value_vars=["train_acc", "validation_acc"],
# 						 var_name="type",
# 						 value_name="acc")

# 		return df_loss, df_acc


# 	def loss_acc_plot(CONTENT):
# 		df_loss, df_acc = get_loss_acc_df(CONTENT)
# 		nearest = alt.selection(type='single', nearest=True, on='mouseover',
# 								fields=['epoch'], empty='none')

# 		loss_line = alt.Chart(df_loss).mark_line(interpolate='basis').encode(
# 			alt.X('epoch:Q', title="Epoch"),
# 			alt.Y('loss:Q', title="Loss"),
# 			alt.Color('type:N', title=""),
# 		)

# 		acc_line = alt.Chart(df_acc).mark_line(interpolate='basis').encode(
# 			alt.X('epoch:Q', title="Epoch"),
# 			alt.Y('acc:Q', title="Accuracy (%)"),
# 			alt.Color('type:N', title=""),
# 		)

# 		selectors = alt.Chart(df_loss).mark_point().encode(
# 			x='epoch:Q',
# 			opacity=alt.value(0),
# 		).add_selection(
# 			nearest
# 		)

# 		loss_points = loss_line.mark_point().encode(
# 			opacity=alt.condition(nearest, alt.value(1), alt.value(0))
# 		)

# 		loss_text = loss_line.mark_text(align='left', dx=5, dy=-5).encode(
# 			text=alt.condition(nearest, 'loss:Q', alt.value(' '))
# 		)

# 		acc_points = acc_line.mark_point().encode(
# 			opacity=alt.condition(nearest, alt.value(1), alt.value(0))
# 		)

# 		acc_text = acc_line.mark_text(align='left', dx=5, dy=-5).encode(
# 			text=alt.condition(nearest, 'acc:Q', alt.value(' '))
# 		)

# 		rules = alt.Chart(df_loss).mark_rule(color='gray').encode(
# 			x='epoch:Q',
# 		).transform_filter(
# 			nearest
# 		)

# 		loss_plot = alt.layer(
# 			loss_line, selectors, loss_points, rules, loss_text
# 		).properties(
# 			width=400, height=200,
# 			title='Train/Validation Loss'
# 		)

# 		acc_plot = alt.layer(
# 			acc_line, selectors, acc_points, rules, acc_text
# 		).properties(
# 			width=400, height=200,
# 			title='Train/Validation Accuracy (%)'
# 		)

# 		return (loss_plot | acc_plot)


# 	def params_plot(CONTENT):
# 		param_df_list, param_df_name = get_param_df(CONTENT)
# 		index_selector = alt.selection(type="single", on='mouseover', fields=['epoch'])
# 		plots = []
# 		for i in range(len(param_df_list)):
# 			p = alt.Chart(param_df_list[i]).mark_rect().encode(
# 				x=alt.X('epoch:O'),
# 				y=alt.Y('value:Q', bin=alt.Bin(maxbins=20)),
# 				# legend=alt.Legend(orient="bottom")
# 				color=alt.Color('count()', legend=None),
# 				opacity=alt.condition(index_selector, alt.value(1.0), alt.value(0.5))
# 			).add_selection(
# 				index_selector
# 			).properties(
# 				width=400, height=200,
# 				title='Model Parameters(' + param_df_name[i] + ')'
# 			)

# 			bar = alt.Chart(param_df_list[0]).mark_bar().encode(
# 				x=alt.X('count()'),
# 				y=alt.Y('value:Q', bin=alt.Bin(maxbins=20), title=''),
# 				# color=alt.Color('blue'),
# 			).transform_filter(
# 				index_selector
# 			).properties(
# 				width=50, height=200,
# 			)

# 			plots.append((p | bar).resolve_scale(
# 				y='shared'
# 			))

# 		return (plots[0] | plots[1]).resolve_scale(
# 					  color='independent'
# 					) & (plots[2] | plots[3]).resolve_scale(
# 					  color='independent'
# 					) & (plots[4] | plots[5]).resolve_scale(
# 					  color='independent'
# 					) & (plots[6] | plots[7]).resolve_scale(
# 					  color='independent'
# 					)
# 	CONTENT = get_content()
# 	st.write(loss_acc_plot(CONTENT))
# 	st.write(params_plot(CONTENT))

if __name__ == "__main__":
	word2vec_dict = load_word2vec_dict(word2vec_urls = ['https://github.com/CMU-IDS-2020/fp-good_or_bad/raw/main/word2vec/100d/word2vec_100d_{}.pt'.format(i+1) for i in range(5)], word2vec_dir = "./word2vec")
	main()