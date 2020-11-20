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

# EMBEDDING_URL = "https://github.com/CMU-IDS-2020/fp-good_or_bad/blob/main/sample_embeddings/sample_words_embeddings"
EMBEDDING_URL = "../sample_words_embeddings"


def main():
	# this should return an input embedding dict that can be put into the run embedding()
	run_preprocess()
	run_embedding()

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

	d = {'Sentiment': ["negative", "somewhat negative", "neutral", "somewhat positive", "positive"], 'Probability': [0.1, 0.2, 0.3, 0.2, 0.2]}
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
	st.write(c)
	st.write("Our model predicts that your input text contains " + max_sentiment + " sentiment!")

def run_embedding(user_input=None):
	@st.cache
	def load_sample_embedding(url):
		sample_embeddings = torch.load(url)
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

	fig = px.scatter_3d(source_3d, x='x', y='y', z='z',color='label', width=1000, height=800, range_x=[-1500,1500], range_y=[-1500,1500], range_z=[-1500,1500])

	fig.update_traces(marker=dict(size=4), selector=dict(mode='markers'))
	fig.update_traces(hovertemplate=' ')
	fig.update_layout(scene_aspectmode='cube')
	st.plotly_chart(fig)

if __name__ == "__main__":
	main()