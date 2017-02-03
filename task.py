import pandas as pd
import tflearn as tf
import numpy as np
import tensorflow
from sklearn.model_selection import train_test_split




from sklearn.feature_extraction.text import CountVectorizer
from tflearn.data_utils import to_categorical, pad_sequences

data_set = pd.read_csv('./ign.csv')

valid_portion = .1
train, test =  train_test_split(data_set, test_size = valid_portion,random_state=12)

def generateX_Y(data):
    X, Y = data.drop('score',1),data['score']
    vectorizer = CountVectorizer(min_df=1)
    bag_trainX = vectorizer.fit_transform(X['title'])

    bag_of_ids_trainX = {"bag": [], 'max_len': bag_trainX.shape[1]}
    for bag in bag_trainX:
        bag_of_ids_trainX['bag'].append(np.where(bag.toarray() > 0)[1])
    bag_of_ids_trainX['bag'] = np.array(pad_sequences(bag_of_ids_trainX['bag'], maxlen=17))

    print([None, int(bag_of_ids_trainX['bag'].shape[1])])
    _, catY = np.unique(data['score_phrase'], return_inverse=True)
    catY = to_categorical(catY, _.shape[0])
    bag_of_ids_trainX['output'] = catY

    #Interests columns
    columns = ['platform', 'genre', 'editors_choice', 'release_year']

    resX = {}
    def toCat(cat):
        d = data[columns]
        _, d= np.unique(d[cat].factorize()[0], return_inverse=True)
        return to_categorical(d,_.shape[0])
    resX['categories'] = [toCat(cat) for cat in columns]
    resX['sentemental'] = bag_of_ids_trainX

    return resX, np.expand_dims(Y.values,axis=1)

#Generate trainX,trainY,testX,testY
trainX, trainY = generateX_Y(train)
testX, testY = generateX_Y(test)


#Create SentamentalAnalys Layer(LSTM)
net = tf.input_data([None, int(trainX['sentemental']['bag'].shape[1])] ,name='input')
net = tf.embedding(net, input_dim=trainX['sentemental']['max_len'], output_dim=512 ,name='embeding')
net = tf.lstm(net, 512, dropout=0.95,name='lstm')
net_sentemental = tf.fully_connected(net,trainY.shape[1],activation='relu' ,name='f_connect')

#Create Layer for each category input, such as : platflorm, genre, editors_choice, release_year
nets = []
for i,prop in enumerate(trainX['categories']):
    net = tf.input_data([None, prop.shape[1]], name='input_column'+str(i))

    output_size = max(int(prop.shape[1] * 0.1),2)
    size = output_size
    while(size > 1):
        net = tf.fully_connected(net,size,activation='relu')
        size /= 2
        size = int(size)

    net = tf.fully_connected(net,output_size ,activation='relu')
    net = tf.dropout(net,0.85)
    nets.append(net)

#Merge Layers for category like merge(merge(merge(Layer1 and Layer2), Layer3),Layer4)
merged_net = nets[0]
for i in range(1,len(nets)):
    net = nets[i]
    merged_net = tf.layers.merge_ops.merge([merged_net,net], mode='concat')
    merged_net = tf.fully_connected(merged_net,5,activation='relu' )


#Merged Layers and Sentemental analys Layer
nets = [merged_net, net_sentemental]
#Merge Layers into one layer
net = tf.layers.merge_ops.merge(nets, mode='concat')

net = tf.fully_connected(net,trainY.shape[1],activation='relu' )
net = tf.regression(net, optimizer='adam', learning_rate=0.00001,name = 'optimizer2',loss='mean_square')

model = tf.DNN(net, tensorboard_verbose=3,checkpoint_path='./checkpoints/check')

def generate_feed_dict(X):
    feed_dict = {'input':X['sentemental']['bag']}

    for i,prop in enumerate(X['categories']):
        feed_dict['input_column'+str(i)] = prop
    return feed_dict
point = tensorflow.train.latest_checkpoint('./checkpoints')

model.fit(generate_feed_dict(trainX), trainY,validation_set=(generate_feed_dict(testX),testY), n_epoch=10000000, show_metric=True,batch_size=32, snapshot_epoch=True, snapshot_step=2000)