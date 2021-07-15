import pprint
import requests
import pyquery
import os


def kernel_1_sample_scrap(
    max_articles=None,
):
    if max_articles is None:
        max_articles = 1

    with requests.get(
        'https://dev.to',
    ) as p:
        t10 = p.content.decode('utf-8')
    t11 = pyquery.PyQuery(t10)
    t13 = t11('.crayons-story__title > a')
    t12 = [
        pyquery.PyQuery(o).attr('href')
        for o in t13
    ]
    pprint.pprint(t12)
    t14 = [
        'https://dev.to/%s' % o
        for o in t12
    ]

    t8 = []
    for t7 in t14[:max_articles]:
        with requests.get(
            t7,
        ) as p:
            t1 = p.content.decode('utf-8')
        t2 = pyquery.PyQuery(t1)
        t3 = t2('.comment__content')
        t6 = []
        for o in t3:
            t4 = pyquery.PyQuery(o)
            t5 = t4('.comment__header > a').attr['href']
            t9 = t4('.comment__body').text()
            t6.append(
                dict(
                    author=t5,
                    text=t9,
                )
            )

        #pprint.pprint(t3)
        pprint.pprint(t6)
        t8.append(
            dict(
                article=t7,
                comments=t6,
            )
        )

    pprint.pprint(t8)

    return dict(
        t1=t1,
        t2=t2,
        t3=t3,
        t6=t6,
        t8=t8,
        t12=t12,
    )

def kernel_2():
    import numpy as np # linear algebra
    import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
    from tqdm import tqdm
    from sklearn.model_selection import train_test_split
    import tensorflow as tf
    from keras.models import Sequential
    from keras.layers.recurrent import LSTM, GRU,SimpleRNN
    from keras.layers.core import Dense, Activation, Dropout
    from keras.layers.embeddings import Embedding
    from keras.layers.normalization import BatchNormalization
    from keras.utils import np_utils
    from sklearn import preprocessing, decomposition, model_selection, metrics, pipeline
    from keras.layers import GlobalMaxPooling1D, Conv1D, MaxPooling1D, Flatten, Bidirectional, SpatialDropout1D
    from keras.preprocessing import sequence, text
    from keras.callbacks import EarlyStopping


    import matplotlib.pyplot as plt
    import seaborn as sns
    #%matplotlib inline
    from plotly import graph_objs as go
    import plotly.express as px
    import plotly.figure_factory as ff

    # %% [markdown]
    # # Configuring TPU's
    #
    # For this version of Notebook we will be using TPU's as we have to built a BERT Model

    # %% [code]
    # Detect hardware, return appropriate distribution strategy
    try:
        # TPU detection. No parameters necessary if TPU_NAME environment variable is
        # set: this is always the case on Kaggle.
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        print('Running on TPU ', tpu.master())
    except ValueError:
        tpu = None

    if tpu:
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.experimental.TPUStrategy(tpu)
    else:
        # Default distribution strategy in Tensorflow. Works on CPU and single GPU.
        strategy = tf.distribute.get_strategy()

    print("REPLICAS: ", strategy.num_replicas_in_sync)

    # %% [code]
    train = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train.csv')
    validation = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/validation.csv')
    test = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/test.csv')

    # %% [markdown]
    # We will drop the other columns and approach this problem as a Binary Classification Problem and also we will have our exercise done on a smaller subsection of the dataset(only 12000 data points) to make it easier to train the models

    # %% [code]
    train.drop(['severe_toxic','obscene','threat','insult','identity_hate'],axis=1,inplace=True)

    # %% [code]
    train = train.loc[:12000,:]
    train.shape

    # %% [markdown]
    # We will check the maximum number of words that can be present in a comment , this will help us in padding later

    # %% [code]
    train['comment_text'].apply(lambda x:len(str(x).split())).max()


    # %% [markdown]
    # ### Data Preparation

    # %% [code]
    xtrain, xvalid, ytrain, yvalid = train_test_split(train.comment_text.values, train.toxic.values,
                                                      stratify=train.toxic.values,
                                                      random_state=42,
                                                      test_size=0.2, shuffle=True)

    # %% [markdown]
    # # Before We Begin
    #
    # Before we Begin If you are a complete starter with NLP and never worked with text data, I am attaching a few kernels that will serve as a starting point of your journey
    # * https://www.kaggle.com/arthurtok/spooky-nlp-and-topic-modelling-tutorial
    # * https://www.kaggle.com/abhishek/approaching-almost-any-nlp-problem-on-kaggle
    #
    # If you want a more basic dataset to practice with here is another kernel which I wrote:
    # * https://www.kaggle.com/tanulsingh077/what-s-cooking
    #
    # Below are some Resources to get started with basic level Neural Networks, It will help us to easily understand the upcoming parts
    # * https://www.youtube.com/watch?v=aircAruvnKk&list=PL_h2yd2CGtBHEKwEH5iqTZH85wLS-eUzv
    # * https://www.youtube.com/watch?v=IHZwWFHWa-w&list=PL_h2yd2CGtBHEKwEH5iqTZH85wLS-eUzv&index=2
    # * https://www.youtube.com/watch?v=Ilg3gGewQ5U&list=PL_h2yd2CGtBHEKwEH5iqTZH85wLS-eUzv&index=3
    # * https://www.youtube.com/watch?v=tIeHLnjs5U8&list=PL_h2yd2CGtBHEKwEH5iqTZH85wLS-eUzv&index=4
    #
    # For Learning how to visualize test data and what to use view:
    # * https://www.kaggle.com/tanulsingh077/twitter-sentiment-extaction-analysis-eda-and-model
    # * https://www.kaggle.com/jagangupta/stop-the-s-toxic-comments-eda

    # %% [markdown]
    # # Simple RNN
    #
    # ## Basic Overview
    #
    # What is a RNN?
    #
    # Recurrent Neural Network(RNN) are a type of Neural Network where the output from previous step are fed as input to the current step. In traditional neural networks, all the inputs and outputs are independent of each other, but in cases like when it is required to predict the next word of a sentence, the previous words are required and hence there is a need to remember the previous words. Thus RNN came into existence, which solved this issue with the help of a Hidden Layer.
    #
    # Why RNN's?
    #
    # https://www.quora.com/Why-do-we-use-an-RNN-instead-of-a-simple-neural-network
    #
    # ## In-Depth Understanding
    #
    # * https://medium.com/mindorks/understanding-the-recurrent-neural-network-44d593f112a2
    # * https://www.youtube.com/watch?v=2E65LDnM2cA&list=PL1F3ABbhcqa3BBWo170U4Ev2wfsF7FN8l
    # * https://www.d2l.ai/chapter_recurrent-neural-networks/rnn.html
    #
    # ## Code Implementation
    #
    # So first I will implement the and then I will explain the code step by step

    # %% [code]
    # using keras tokenizer here
    token = text.Tokenizer(num_words=None)
    max_len = 1500

    token.fit_on_texts(list(xtrain) + list(xvalid))
    xtrain_seq = token.texts_to_sequences(xtrain)
    xvalid_seq = token.texts_to_sequences(xvalid)

    #zero pad the sequences
    xtrain_pad = sequence.pad_sequences(xtrain_seq, maxlen=max_len)
    xvalid_pad = sequence.pad_sequences(xvalid_seq, maxlen=max_len)

    word_index = token.word_index

    # %% [code]
    #%%time
    with strategy.scope():
        # A simpleRNN without any pretrained embeddings and one dense layer
        model = Sequential()
        model.add(Embedding(len(word_index) + 1,
                         300,
                         input_length=max_len))
        model.add(SimpleRNN(100))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.summary()

    return dict(
        model=model,
        xtrain_pad=xtrain_pad,
        strategy=strategy,
        xvalid_pad=xvalid_pad,
        xtrain_seq=xtrain_seq,
        token=token,
        max_len=max_len,
        xtrain=xtrain,
        xvalid=xvalid,
        ytrain=ytrain,
        yvalid=yvalid,
    )


def kernel_3(
    o_2,
    nb_epochs=None,
):
    if nb_epochs is None:
        nb_epochs = 5

    # %% [markdown]
    # Writing a function for getting auc score for validation

    # %% [code]
    def roc_auc(predictions,target):
        import sklearn.metrics
        '''
        This methods returns the AUC Score when given the Predictions
        and Labels
        '''

        fpr, tpr, thresholds = sklearn.metrics.roc_curve(target, predictions)
        roc_auc = sklearn.metrics.auc(fpr, tpr)
        return roc_auc

    # %% [code]
    if os.path.exists('model.h5'):
        o_2['model'].load_weights('model.h5')
    else:
        o_2['model'].fit(
            o_2['xtrain_pad'],
            o_2['ytrain'],
            nb_epoch=nb_epochs,
            batch_size=64*o_2['strategy'].num_replicas_in_sync
        ) #Multiplying by Strategy to run on TPU's
        o_2['model'].save_weights('model.h5')

    # %% [code]
    scores = o_2['model'].predict(o_2['xvalid_pad'])
    print(
        "Auc: %.2f%%" % (
            roc_auc(
                scores,
                o_2['yvalid']
            )
        )
    )

    # %% [code]
    scores_model = []
    scores_model.append(
        {
            'Model': 'SimpleRNN',
            'AUC_Score': roc_auc(
                scores,
                o_2['yvalid']
            )
        }
    )

    # %% [markdown]
    # ## Code Explanantion
    # * Tokenization<br><br>
    #  So if you have watched the videos and referred to the links, you would know that in an RNN we input a sentence word by word. We represent every word as one hot vectors of dimensions : Numbers of words in Vocab +1. <br>
    #   What keras Tokenizer does is , it takes all the unique words in the corpus,forms a dictionary with words as keys and their number of occurences as values,it then sorts the dictionary in descending order of counts. It then assigns the first value 1 , second value 2 and so on. So let's suppose word 'the' occured the most in the corpus then it will assigned index 1 and vector representing 'the' would be a one-hot vector with value 1 at position 1 and rest zereos.<br>
    #   Try printing first 2 elements of xtrain_seq you will see every word is represented as a digit now

    # %% [code]
    o_2['xtrain_seq'][:1]

def kernel_4(
    o_2,
    input_texts=None,
):
    import keras.preprocessing.sequence

    if input_texts is None:
        input_texts = [
            'blahb blahb blah',
            'Hello World!',
            'This is very good!',
            'A very non toxic comment! This is so polite and polished one!'
        ]

    t6 = []
    for o in input_texts:
        t1 = o
        t2 = o_2['token'].texts_to_sequences(
            [t1],
        )
        t3 = keras.preprocessing.sequence.pad_sequences(
            t2,
            maxlen=o_2['max_len']
        )
        t4 = o_2['model'].predict(
            t3,
        )
        t6.append(
            dict(
                text=o,
                score=t4[0][0],
            )
        )
        pprint.pprint(
            dict(
                t1=t1,
                t2=t2,
                t3=t3,
                t4=t4,
            )
        )
    pprint.pprint(t6)

    return dict(
        t6=t6,
    )

def kernel_5(
    o_1=None,
    o_2=None,
):
    if o_1 is None:
        o_1 = kernel_1_sample_scrap(max_articles=5)

    if o_2 is None:
        o_2 = kernel_2()
        o_3 = kernel_3(
            o_2=o_2,
            nb_epochs=1
        )

    t1 = sum(
        [
            [
                o['text'] for o in o2['comments']
            ] for o2 in o_1['t8']
        ], []
    )

    t2 = kernel_4(
        o_2=o_2,
        input_texts=t1
    )

    t3 = sorted(
        t2['t6'],
        key=lambda x: x['score'],
    )
    pprint.pprint(t3)
