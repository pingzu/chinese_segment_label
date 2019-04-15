import pickle
import time
import sys
import numpy as np
from config import config
from utility import *
import os

def load_data(kind):
    X, y = load_pickle(config[kind]["processed_data"])

    # 保存当前状态，使两次产生的随机序列，以同序打乱
    state = np.random.get_state()
    np.random.shuffle(X)
    np.random.set_state(state)
    np.random.shuffle(y)
        
    vlen = int(len(X) * 0.3)
    tlen = int(len(X) * 1)

    trainX, valX = X[:tlen], X[-vlen:]
    trainy, valy = y[:tlen], y[-vlen:]

    return trainX, trainy, valX, valy

def train_model(kind):
    print('train model', kind)
    tX, ty, vX, vy= load_data(kind)

    model = create_model_by_kind(kind)
    batch_size = 1024
    epochs = config[kind]["epochs"]
    history = model.fit(
        x=tX,
        y=ty,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(vX,vy)
    )
    model.save(config[kind]["model"])
    save_pickle(config[kind]['history'],history.history)


# def train_segment():
#     print('train_segment')
#     tX, ty, vX, vy= load_data(kind='segment')
#     # exit()
#     model = create_model_by_kind(kind='segment')
#     batch_size = 1024
#     epochs = config['segment']["epochs"]
#     history = model.fit(
#         x=tX,
#         y=ty,
#         batch_size=batch_size,
#         epochs=epochs,
#         validation_data=(vX,vy)
#     )

#     s = config['segment']['sequence_length']
#     d = config['segment']['word_dimension']
#     u = config['segment']['hidden_units']

#     tid = 's{}-d{}-u{}-e{}'.format(s,d,u,epochs)
#     save_pickle('data/temp/history-'+tid,history.history)
#     try:
#         model.save('data/temp/segmodel-'+tid)
#     except:
#         model.save('data/temp/segmodel-'+tid)


# def train_label():
#     print('train_label')
#     tX, ty, vX, vy= load_data(kind='label')

#     model = create_model_by_kind(kind='label')
#     batch_size = 1024

#     epochs = config['label']["epochs"]
#     history = model.fit(
#         x=tX,
#         y=ty,
#         batch_size=batch_size,
#         epochs=epochs,
#         validation_data=(vX,vy)
#     )

#     save_pickle(config['label']["history"],history.history)
#     model.save(config['label']["model"])


# def train_param():
#     tX, ty, vX, vy= load_data(kind='segment')

#     sequence = [config['segment']['sequence_length']]
#     # sequence = [16, 24, 32, 48, 64]
#     dimension = [128, 256, 384, 512]
#     units = [128, 256, 512]
#     parms = [(1024,256),(1024,512)]
#     batch_size = 2048
#     epochs = 15

#     for s in sequence:
#         for d , u in parms:
#             train_id = 's{}-d{}-u{}-e{}'.format(s,d,u,epochs)
#             print('start training: sequence_length={},word_dimension={},hidden_units={},epochs={}'.format(s,d,u,epochs))
#             model = create_train_model(sequence_length=s,word_dimension=d,hidden_units=u)

#             history = model.fit(
#                 x=tX,
#                 y=ty,
#                 batch_size=batch_size,
#                 epochs=epochs,
#                 validation_data=(vX,vy)
#             )
#             model_path = 'data/temp/segmodel-{}'.format(train_id)
#             history_path = 'data/temp/history-{}'.format(train_id)
#             try:
#                 model.save(model_path)
#             except:
#                 model.save(model_path+'_')

#             # history.save(history_path)
#             save_pickle(history_path,history.history)



def main():
    if 'segment' in sys.argv:
        train_model('segment')
    if 'label' in sys.argv:    
        train_model('label')


if __name__ == "__main__":
    start = time.clock()
    main()
    # plot_history()
    # train_param()
    elapsed = (time.clock() - start)
    print("Time used:", elapsed)
   