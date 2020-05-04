# -*- coding: utf-8 -*-
import keras
import numpy as np
from keras.models import Sequential,load_model
from keras.layers import Dense,Activation,Dropout
from neural_network.draw_curve import draw_curve,plot_history

    #建立模型,rmsprop优化器，mse损失函数，MAE衡量模型指标
def build_model(in_size,optimizer_name='rmsprop',loss_name='mse',metrics_name=['mae']):
    model = Sequential()
    #输入层
    model.add(Dense(16,activation='relu',input_dim=in_size))

    #第一个隐含层
    model.add(Dropout(0.3,noise_shape=None,seed=None))
    model.add(Dense(16,activation='relu',input_dim=in_size))

    #第二个隐含层
    model.add(Dropout(0.2, noise_shape=None, seed=None))
    model.add(Dense(8,activation='relu'))

    #第三个隐含层
    '''
    model.add(Dropout(0.2, noise_shape=None, seed=None))
    model.add(Dense(64,activation='relu'))

    model.add(Dropout(0.1, noise_shape=None, seed=None))
    model.add(Dense(32, activation='relu'))
    '''
    #输出
    model.add(Dense(1))
    model.compile(optimizer=optimizer_name,loss=loss_name,metrics=metrics_name)
    return model

# Display training progress by printing a single dot for each completed epoch
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')

def train_model(train_data,train_targets):
        model = build_model(train_data.shape[1])
        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)
        history = model.fit(train_data, train_targets,epochs=1500,
                    validation_split=0.2, verbose=1)#callbacks=[early_stop,PrintDot()]
        model.save('wod_model.h5')
        model = load_model('wod_model.h5')
        plot_history(history)
        return history,model

    #k折验证
def K_fold_verfication(train_data,train_targets,test_data,test_targets,k=3,epochs=100):
    num_val_samples = int(len(train_data)/k) #把数据分成k份，计算每一份的数量
    num_epochs = epochs #通过神经网络的次数
    all_scores = []

    #分区进行训练验证
    for i in range(k):
        val_data=train_data[i*num_val_samples:(i+1)*num_val_samples]
        val_targets = train_targets[i*num_val_samples:(i+1)*num_val_samples]

        partial_train_data = np.concatenate([train_data[:i*num_val_samples],train_data[(i+1)*num_val_samples:]],axis=0)#和其他分区的数据组合
        partial_train_targets = np.concatenate([train_targets[:i*num_val_samples],train_targets[(i+1)*num_val_samples:]],axis=0)

        #掩膜处理
        mask_array1 = partial_train_data == -9999
        partial_train_data = np.ma.array(partial_train_data, mask=mask_array1)
        mask_array2 = partial_train_targets == -9999
        partial_train_targets = np.ma.array(partial_train_targets, mask=mask_array2)
        mask_array3 = val_data == -9999
        val_data = np.ma.array(val_data, mask=mask_array3)
        mask_array4 = val_targets == -9999
        val_targets = np.ma.array(val_targets, mask=mask_array4)

        # 加载模型
        #model = load_model('my_model.h5')
        model=build_model(train_data.shape[1])

        #训练模型（静默模式）
        result = model.fit(partial_train_data,partial_train_targets,validation_data=(val_data,val_targets),epochs=num_epochs,batch_size=1,verbose=0)

        #先用验证数据集评估模型
        val_mse,val_mae = model.evaluate(val_data,val_targets,verbose=0)
        print(np.mean(val_mse),np.mean(val_mae))
        #all_scores.append(val_mae)

        #在所有训练数据上训练模型
        mae_history = result.history['mae']
        #val_mae_history = result.history['val_mae']
        all_scores.append(mae_history)
        print(np.mean(all_scores))

    #绘制验证分数

    #average_mae_history = [np.mean([x[i] for x in all_scores]) for i in range(num_epochs)]
    #mae_history_list = draw_curve(average_mae_history)
    #epochs_min = mae_history_list.index(min(mae_history_list))

    model.save('my_model.h5')

    model = load_model('my_model.h5')
    model.fit(train_data,train_targets,epochs=epochs_min,batch_size=1,verbose=1)
    test_mse_score, test_mae_score = model.evaluate(test_data, test_targets)
    print(test_mse_score,test_mae_score)
    return all_scores

#def save_result(train_data,train_targets):





    #将数据标准化处理，标准差为1
def data_standardization(pre_data):
    mean_data = np.mean(pre_data,axis=0)
    data_vias = pre_data - mean_data

    std_data = np.std(data_vias,axis=0)
    return data_vias/std_data

    #数据归一化
def data_normalization(pre_data):
    data_max = np.max(pre_data)
    data_min = np.min(pre_data)
    result = (pre_data-data_min)/(data_max-data_min)
    return result