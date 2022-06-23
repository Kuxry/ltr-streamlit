
import streamlit as st
import pandas as pd
import altair as alt


# Add a selectbox to the sidebar:
from evaluation import evaluation

st.set_page_config(layout="wide", page_title="Learning To Rank")

dataset, loss_function, evaluation1 = st.columns(3)
with dataset:
    st.sidebar.header("Which dataset do you want use?")
    dataset_mode = st.sidebar.selectbox("Choose the mode",
                                        ["Microsoft", "Yahoo"])

with loss_function:
    st.sidebar.header("Which loss function do you want use?")
    lf_mode = st.sidebar.selectbox("Choose the mode",
                                   ["RankMSE", "RankNet", "LambdaRank"])

with evaluation1:
    st.sidebar.header("Which evaluation do you want use?")
    eva_mode = st.sidebar.selectbox("Choose the mode",
                                    ["NDCG", "Other"])

st.sidebar.title("Batch size")
batchsize = st.sidebar.slider('', 1, 100)
# st.write(x, 'squared is', x * x)
st.sidebar.write(batchsize)

# 正文
st.markdown("<h1 style='text-align: center; color: black;'>Welcome to Learning To Rank👋</h1>", unsafe_allow_html=True)
# 可扩展段落
st.write("")
st.markdown(
    """

    **Learning to Rank** is a machine Learning model. It uses machine learning methods, 
    we can take the [output as feature] of each existing ranking model, and then train a new model, and automatically learn the parameters of the new model, 
    so it is very convenient to combine multiple existing ranking model to generate a new ranking model.

    **👈 Select a way you want to try from the dropdown on the left** to see some results
    of what Learning To Rank can do!

    ### What is the dataset?

    """
)
with st.expander("Mircosoft"):
    st.markdown("""
        **LETOR(Learning to Rank for Information Retrieval)** is a package of benchmark data sets for research on Learning To Rank, 
        which contains standard features, relevance judgments, data partitioning, evaluation tools, and several baselines. 

        There are about 1700 queries in MQ2007 with labeled documents and about 800 queries in MQ2008 with labeled documents.

        ### Datasets
        The 5-fold cross validation strategy is adopted and the 5-fold partitions are included in the package. 
        In each fold, there are three subsets for learning: training set, validation set and testing set.
        ##### Descriptions
        Each row is a query-document pair. The first column is **relevance label** of this pair, 
        the second column is **query id**, the following columns are **features**, and the end of the row is **comment** about the pair, including id of the document.
        **The larger the relevance label, the more relevant the query-document pair.**
        A query-document pair is represented by a 46-dimensional feature vector. 

        Here are several example rows from MQ2007 dataset:

        ----
        > 2 qid:10032 1:0.056537 2:0.000000 3:0.666667 4:1.000000 5:0.067138 … 45:0.000000 46:0.076923 #docid = GX029-35-5894638 inc = 0.0119881192468859 prob = 0.139842

        > 0 qid:10032 1:0.279152 2:0.000000 3:0.000000 4:0.000000 5:0.279152 … 45:0.250000 46:1.000000 #docid = GX030-77-6315042 inc = 1 prob = 0.341364

        > 0 qid:10032 1:0.130742 2:0.000000 3:0.333333 4:0.000000 5:0.134276 … 45:0.750000 46:1.000000 #docid = GX140-98-13566007 inc = 1 prob = 0.0701303

        > 1 qid:10032 1:0.593640 2:1.000000 3:0.000000 4:0.000000 5:0.600707 … 45:0.500000 46:0.000000 #docid = GX256-43-0740276 inc = 0.0136292023050293 prob = 0.400738



        """)

with st.expander("Yahoo"):
    st.markdown("""

        """)

st.markdown(
    """

    - Mircosoft dataset
     [LETOR: Learning to Rank for Information Retrieval](https://www.microsoft.com/en-us/research/project/letor-learning-rank-information-retrieval/)
    - Yahoo dataset
     [Yahoo](https://github.com/streamlit/demo-uber-nyc-pickups)
    """
)

st.markdown(
    """
    ### What you can get?
    - The NDCG score of each fold
    - The elapsed time of five folds
    - The NDCG average score of five folds

    - Displays changes in NDCG Average Score

    """
)

st.title("OUTPUT")
Result, NDCG_score = st.columns(2)
with Result:
    # split_type=SPLIT_TYPE.Train
    st.subheader("Result")
    # train_data = load_data(data_id=data_id, file=file_train, split_type=split_type, batch_size=batch_size)
    # # st.write(train_data.batch_size)
    #
    # for batch_ids, batch_q_doc_vectors, batch_std_labels in train_data:
    #     st.write(batch_q_doc_vectors.shape)
    data_id = 'MQ2008_Super'
    dir_data = '/home/uba/data/MQ2008/'
    average = evaluation(data_id=data_id, dir_data=dir_data, model_id=lf_mode, batch_size=batchsize)

st.subheader(" ")
with NDCG_score:
    st.subheader("NDCG Score")
    pb = st.progress(0)
    status_txt = st.empty()

    for i in range(100):
        pb.progress(i)
        new_rows = average
        status_txt.text(
            "The 5-fold average scores : %s" % new_rows
        )

    # 折线图
    source_average = pd.DataFrame(average,
                                  columns=['NDCG Average Scores'], index=pd.RangeIndex(start=1, stop=11, name='x'))
    source_average = source_average.reset_index().melt('x', var_name='Average Scores', value_name='y')

    line_chart = alt.Chart(source_average).mark_line(interpolate='basis').encode(
        alt.X('x', title='The Number Of DCG'),
        alt.Y('y', title='NDCG value'),
        color='Average Scores:N'
    ).properties(
        title='NDCG Average Score Trends'
    )
    st.altair_chart(line_chart)

    st.subheader("Displays changes in NDCG Average Score")
    for i in range(10):
        st.metric(label="nDCG average scores", value=average[i], delta=average[i] - average[i - 1])
