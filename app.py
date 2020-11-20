import streamlit as st
import numpy as np
import pandas as pd
import torch

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

MODEL_PATH = 'https://github.com/CMU-IDS-2020/fp-good_or_bad/raw/main/models/xentropy_adam_lr0.0001_wd0.0005_bs128'
EMBEDDING_URL = "https://github.com/CMU-IDS-2020/fp-good_or_bad/raw/main/sample_embeddings/sample_words_embeddings.pt"
MODEL_PATH_PT = 'https://github.com/CMU-IDS-2020/fp-good_or_bad/raw/main/models/xentropy_adam_lr0.0001_wd0.0005_bs128.pt'
EPOCH = 50
SAMPLE_LIMIT = 5000
EPOCH_SAMPLE_LIMIT = SAMPLE_LIMIT // EPOCH


def main():
	# we should return an input embedding dict that can be put into the run embedding()

	preprocessed = run_preprocess()
	run_train()
	run_embedding()
	run_predict(preprocessed)


def run_preprocess():
	# tokenize -> lowercase -> remove stopwords -> lemmatize
	def tokenize_text(text):
		tokenizer = RegexpTokenizer(r'\w+')
		return tokenizer.tokenize(text)

	def lowercase_text(tokens):
		return [token.lower() for token in tokens]

	def remove_stopwords(tokens):
		english_stopwords = stopwords.words('english')
		return [token if token not in english_stopwords else None for token in tokens]

	def lemmatize(tokens):
		lemmatizer = WordNetLemmatizer()
		return [lemmatizer.lemmatize(token) if token else None for token in tokens]

	st.title("Predict Sentiment")
	input = st.text_input('Write something emotional and hit enter!', 'I absolutely love it!')
	tokens = tokenize_text(input)
	lowercase_tokens = lowercase_text(tokens)
	removed_stopwords = remove_stopwords(lowercase_tokens)
	lemmatized = lemmatize(removed_stopwords)
	g = Digraph()
	i = 0
	g.node(input)
	for token, lc_token, r_token, l_token in zip(tokens, lowercase_tokens, removed_stopwords, lemmatized):
		g.node(token + "token" + str(i), label=token)
		g.edge(input, token + "token" + str(i))
		g.node(lc_token + "lc_token" + str(i), label=lc_token)
		g.edge(token + "token" + str(i), lc_token + "lc_token" + str(i))
		if r_token:
			g.node(r_token + "r_token" + str(i), label=r_token)
			g.edge(lc_token + "lc_token" + str(i), r_token + "r_token" + str(i))
			g.node(l_token + "l_token" + str(i), label=l_token)
			g.edge(r_token + "r_token" + str(i), l_token + "l_token" + str(i))
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
			c.node(token + "token" + str(i))
		c.attr(label='Word Tokens')

	with g.subgraph(name='cluster_3') as c:
		c.attr(color='white')
		c.node_attr['style'] = 'filled'
		for i, token in enumerate(lowercase_tokens):
			c.node(token + "lc_token" + str(i))
		c.attr(label='Lowercase Tokens')

	with g.subgraph(name='cluster_4') as c:
		c.attr(color='white')
		c.node_attr['style'] = 'filled'
		for i, token in enumerate(removed_stopwords):
			if token:
				c.node(token + "r_token" + str(i))
		c.attr(label='Stopwords Removed')

	with g.subgraph(name='cluster_5') as c:
		c.attr(color='white')
		c.node_attr['style'] = 'filled'
		for i, token in enumerate(lemmatized):
			if token:
				c.node(token + "l_token" + str(i))
		c.attr(label='Lemmatized Tokens')

	st.graphviz_chart(g)
	return [token for token in lemmatized]


def run_predict(input):
	@st.cache
	def get_model():
		model = Network(input_channel, out_channel, kernel_sizes, output_dim)
		model.load_state_dict(
			torch.hub.load_state_dict_from_url(MODEL_PATH_PT, progress=False, map_location=torch.device('cpu')))
		return model


	# model = get_model()
	# probs = model()
	d = {'Sentiment': ["negative", "somewhat negative", "neutral", "somewhat positive", "positive"],
		 'Probability': [0.1, 0.2, 0.3, 0.2, 0.2]}
	max_sentiment = d["Sentiment"][np.argmax(d["Probability"])]
	source = pd.DataFrame(d)
	c = alt.Chart(source).mark_bar().encode(
		alt.X('Probability:Q', axis=alt.Axis(format='.0%')),
		alt.Y('Sentiment:N', sort=d['Sentiment']),
		color=alt.condition(
			alt.datum.Sentiment == max_sentiment,  # If the year is 1810 this test returns True,
			alt.value('orange'),  # which sets the bar orange.
			alt.value('steelblue')  # And if it's not true it sets the bar steelblue.
		)
	)
	st.write(c)
	st.write("Our model predicts that your input text contains " + max_sentiment + " sentiment!")


def run_embedding(user_input=None):
	@st.cache
	def load_sample_embedding(url):
		embedding_path = "embedding"
		torch.hub.download_url_to_file(url, embedding_path)
		sample_embeddings = pickle.load(open(embedding_path, "rb"))
		tokens = []
		labels = []
		shapes = []

		for key, val in sample_embeddings.items():
			tokens.append(val)
			labels.append(key)
			shapes.append(0)
		return tokens, labels, shapes

	@st.cache
	def load_usr_embedding(input_dict, tokens, labels, shapes):
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

	'''
	TODO: color label according to negative/positive of emotion?
		  Show text for each point
		  use different shape for user input
	'''

	tokens, labels, shapes = load_sample_embedding(EMBEDDING_URL)
	if user_input is not None:
		tokens, labels, shapes = load_usr_embedding(input_dict, tokens, labels, shapes)
	values_3d = transform_3d(tokens)
	source_3d = get_df(values_3d, labels, shapes)

	fig = px.scatter_3d(source_3d, x='x', y='y', z='z', color='label', width=1000, height=800, range_x=[-1500, 1500],
						range_y=[-1500, 1500], range_z=[-1500, 1500])

	fig.update_traces(marker=dict(size=4), selector=dict(mode='markers'))
	fig.update_traces(hovertemplate=' ')
	fig.update_layout(scene_aspectmode='cube')
	st.plotly_chart(fig)


def run_train():
	@st.cache
	def get_content():
		return torch.hub.load_state_dict_from_url(MODEL_PATH, progress=False, map_location=torch.device('cpu'))

	def get_param_df(content):
		model_parameters = content['model_parameters']

		param_df = pd.DataFrame({'epoch': [], 'param_type': [], 'value': []},
								columns=['epoch', 'param_type', 'value'])

		for i in range(len(model_parameters)):
			epoch = i + 1
			params = model_parameters[i]
			for key in params.keys():
				param_type = key
				values = params[key].numpy().reshape(-1).tolist()
				if len(values) > EPOCH_SAMPLE_LIMIT:
					values = sample(values, EPOCH_SAMPLE_LIMIT)
				rows = pd.DataFrame(
					{'epoch': [epoch] * len(values), 'param_type': [param_type] * len(values), 'value': values})
				param_df = param_df.append(rows, ignore_index=True)

		# convs.0.weight
		# convs.0.bias
		# convs.1.weight
		# convs.1.bias
		# convs.2.weight
		# convs.2.bias
		# linear.weight
		# linear.bias
		convs_0_weight_df = param_df[param_df['param_type'] == 'convs.0.weight']
		convs_0_bias_df = param_df[param_df['param_type'] == 'convs.0.bias']
		convs_1_weight_df = param_df[param_df['param_type'] == 'convs.1.weight']
		convs_1_bias_df = param_df[param_df['param_type'] == 'convs.1.bias']
		convs_2_weight_df = param_df[param_df['param_type'] == 'convs.2.weight']
		convs_2_bias_df = param_df[param_df['param_type'] == 'convs.2.bias']
		linear_weight_df = param_df[param_df['param_type'] == 'linear.weight']
		linear_bias_df = param_df[param_df['param_type'] == 'linear.bias']

		param_df_list = [convs_0_weight_df, convs_0_bias_df, convs_1_weight_df, convs_1_bias_df, convs_2_weight_df,
						 convs_2_bias_df, linear_weight_df, linear_bias_df]
		param_df_name = ["convs.0.weight", "convs.0.bias", "convs.1.weight", "convs.1.bias", "convs.2.weight",
						 "convs.2.bias", "linear.weight", "linear.bias"]
		return param_df_list, param_df_name

	def get_loss_acc_df(content):
		train_loss = content['train_loss']
		train_acc = content['train_acc']
		validation_loss = content['test_loss']
		validation_acc = content['test_acc']
		avg_train_time = content['ave_train_time']

		df = pd.DataFrame({'train_loss': train_loss, 'train_acc': train_acc, 'validation_loss': validation_loss,
						   'validation_acc': validation_acc, 'epoch': range(1, EPOCH + 1)},
						  columns=['train_loss', 'train_acc', 'validation_loss', 'validation_acc', 'epoch'])

		df_loss = df.melt(id_vars=["epoch"],
						  value_vars=["train_loss", "validation_loss"],
						  var_name="type",
						  value_name="loss")
		df_acc = df.melt(id_vars=["epoch"],
						 value_vars=["train_acc", "validation_acc"],
						 var_name="type",
						 value_name="acc")

		return df_loss, df_acc

	def loss_acc_plot(CONTENT):
		df_loss, df_acc = get_loss_acc_df(CONTENT)
		nearest = alt.selection(type='single', nearest=True, on='mouseover',
								fields=['epoch'], empty='none')

		loss_line = alt.Chart(df_loss).mark_line(interpolate='basis').encode(
			alt.X('epoch:Q', title="Epoch"),
			alt.Y('loss:Q', title="Loss"),
			alt.Color('type:N', title=""),
		)

		acc_line = alt.Chart(df_acc).mark_line(interpolate='basis').encode(
			alt.X('epoch:Q', title="Epoch"),
			alt.Y('acc:Q', title="Accuracy (%)"),
			alt.Color('type:N', title=""),
		)

		selectors = alt.Chart(df_loss).mark_point().encode(
			x='epoch:Q',
			opacity=alt.value(0),
		).add_selection(
			nearest
		)

		loss_points = loss_line.mark_point().encode(
			opacity=alt.condition(nearest, alt.value(1), alt.value(0))
		)

		loss_text = loss_line.mark_text(align='left', dx=5, dy=-5).encode(
			text=alt.condition(nearest, 'loss:Q', alt.value(' '))
		)

		acc_points = acc_line.mark_point().encode(
			opacity=alt.condition(nearest, alt.value(1), alt.value(0))
		)

		acc_text = acc_line.mark_text(align='left', dx=5, dy=-5).encode(
			text=alt.condition(nearest, 'acc:Q', alt.value(' '))
		)

		rules = alt.Chart(df_loss).mark_rule(color='gray').encode(
			x='epoch:Q',
		).transform_filter(
			nearest
		)

		loss_plot = alt.layer(
			loss_line, selectors, loss_points, rules, loss_text
		).properties(
			width=400, height=200,
			title='Train/Validation Loss'
		)

		acc_plot = alt.layer(
			acc_line, selectors, acc_points, rules, acc_text
		).properties(
			width=400, height=200,
			title='Train/Validation Accuracy (%)'
		)

		return (loss_plot | acc_plot).resolve_scale(
			color='independent'
		)

	def params_plot(CONTENT):
		param_df_list, param_df_name = get_param_df(CONTENT)
		index_selector = alt.selection(type="single", on='mouseover', fields=['epoch'])
		plots = []
		for i in range(len(param_df_list)):
			p = alt.Chart(param_df_list[i]).mark_rect().encode(
				x=alt.X('epoch:O'),
				y=alt.Y('value:Q', bin=alt.Bin(maxbins=20)),
				# legend=alt.Legend(orient="bottom")
				color=alt.Color('count()', legend=None),
				opacity=alt.condition(index_selector, alt.value(1.0), alt.value(0.5))
			).add_selection(
				index_selector
			).properties(
				width=400, height=200,
				title='Model Parameters(' + param_df_name[i] + ')'
			)

			bar = alt.Chart(param_df_list[0]).mark_bar().encode(
				x=alt.X('count()'),
				y=alt.Y('value:Q', bin=alt.Bin(maxbins=20), title=''),
				# color=alt.Color('blue'),
			).transform_filter(
				index_selector
			).properties(
				width=50, height=200,
			)

			plots.append((p | bar).resolve_scale(
				y='shared'
			))

		return (plots[0] | plots[1]).resolve_scale(
			color='independent'
		) & (plots[2] | plots[3]).resolve_scale(
			color='independent'
		) & (plots[4] | plots[5]).resolve_scale(
			color='independent'
		) & (plots[6] | plots[7]).resolve_scale(
			color='independent'
		)

	CONTENT = get_content()
	st.write(loss_acc_plot(CONTENT))
	st.write(params_plot(CONTENT))


if __name__ == "__main__":
	main()