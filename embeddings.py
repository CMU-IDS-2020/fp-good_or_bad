import streamlit as st
import numpy as np
import pandas as pd
import torch
from sklearn.manifold import TSNE
import plotly.express as px

# EMBEDDING_URL = "https://github.com/CMU-IDS-2020/fp-good_or_bad/blob/main/sample_embeddings/sample_words_embeddings"
EMBEDDING_URL = "../sample_words_embeddings"


def main():
	run_embedding()


def run_embedding():
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
	# tokens, labels, shapes = load_usr_embedding(input_dict, tokens, labels, shapes)
	values_3d = transform_3d(tokens)
	source_3d = get_df(values_3d, labels, shapes)

	fig = px.scatter_3d(source_3d, x='x', y='y', z='z',color='label', width=1000, height=800, range_x=[-1500,1500], range_y=[-1500,1500], range_z=[-1500,1500])

	fig.update_traces(marker=dict(size=4), selector=dict(mode='markers'))
	fig.update_traces(hovertemplate=' ')
	fig.update_layout(scene_aspectmode='cube')
	st.plotly_chart(fig)

if __name__ == "__main__":
	main()