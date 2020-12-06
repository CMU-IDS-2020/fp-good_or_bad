import pandas as pd
import altair as alt
from random import sample
import streamlit as st
import torch

EPOCH = 50
SAMPLE_LIMIT = 5000
EPOCH_SAMPLE_LIMIT = SAMPLE_LIMIT // EPOCH

@st.cache(allow_output_mutation=True)
# dataset_path = "amazon_products" or "movie_reviews" or "yelp_restaurants"
# optimizer_path = "xentropy_adam_all" or "xentropy_sgdmomentum_all"
def get_train_content(dataset_path, optimizer_path):
    url = "https://github.com/CMU-IDS-2020/fp-good_or_bad/raw/main/models/{}/{}".format(dataset_path, optimizer_path)
    return torch.hub.load_state_dict_from_url(url, progress=False, map_location=torch.device('cpu'))

def get_param_df(content):
    param_df = content['model_parameters']

    # param_df = pd.DataFrame({'epoch': [], 'param_type': [], 'value': []},
    #                   columns=['epoch', 'param_type', 'value'])

    # for i in range(len(model_parameters)):
    #     epoch = i+1
    #     params = model_parameters[i]
    #     for key in params.keys():
    #         param_type = key
    #         values = params[key].numpy().reshape(-1).tolist()
    #         if len(values) > EPOCH_SAMPLE_LIMIT:
    #             values = sample(values, EPOCH_SAMPLE_LIMIT)
    #         rows = pd.DataFrame({'epoch': [epoch]*len(values), 'param_type': [param_type]*len(values), 'value': values})
    #         param_df = param_df.append(rows, ignore_index=True)

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

    param_df_list = [convs_0_weight_df, convs_0_bias_df, convs_1_weight_df, convs_1_bias_df, convs_2_weight_df, convs_2_bias_df, linear_weight_df, linear_bias_df]
    param_df_name = ["convs.0.weight", "convs.0.bias", "convs.1.weight", "convs.1.bias", "convs.2.weight", "convs.2.bias", "linear.weight", "linear.bias"]
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

    return (loss_plot | acc_plot)


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

        plots.append((p | bar))

    return (plots[0] | plots[1]).resolve_scale(
          color='independent'
        ) & (plots[2] | plots[3]).resolve_scale(
          color='independent'
        ) & (plots[4] | plots[5]).resolve_scale(
          color='independent'
        ) & (plots[6] | plots[7]).resolve_scale(
          color='independent'
        )
