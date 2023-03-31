#from keras.models import Model
#from keras.layers import Input, Dense, LSTM, Bidirectional
#from keras import backend as K
#from keras import backend as K

from tensorflow.keras import backend as K
from tensorflow.keras import Input
from tensorflow.keras.layers import *   # LSTM, Bidirectional, Dense
from tensorflow.keras.models import load_model, Model
from tensorflow.keras import Model, regularizers
from sklearn.model_selection import train_test_split
from _utils.attention import Attention
import sys
from sklearn.utils import class_weight
from keract import get_activations
from sklearn.model_selection import StratifiedKFold

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import Callback


from sklearn.metrics import *
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import classification_report

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# pd.set_option('display.max_colwidth', -1)  #각 컬럼 width 최대로 
# pd.set_option('display.max_rows', 50)      # display 50개 까지 

#data_dir = '/home/taehyun/Project/cardio/data/'



class Auto_lstm_attention():

    def __init__(self):
        a=1
        self.a=a
        

    def train_test_split(self, x, y, del_flag=True):
        x_train_person, x_test_person, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.3, random_state=1) 
        # random_state=None, default=None
        print(x_train_person)
        
        if del_flag :
            # 0=person_id
            # 2=label
            x_train=np.delete(x_train_person,0,2)
            x_train=np.delete(x_train,2,2)

            x_test=np.delete(x_test_person,0,2)
            x_test=np.delete(x_test,2,2)
        else : 
            x_train = x_train_person
            x_test = x_test_person
        
        ## 훈련데이터세트의 shape ##
        print("RAW가 2개 더 많은게 맞음. x shape", x.shape)
        ## 훈련세트, 테스트세트 shape ##
        print("x_train shape:", x_train.shape,"\n","x_test shape:",x_test.shape)
        ## 훈련데이터세트의 클래스 비율 ##
        print(np.unique(y, return_counts = True))
        ## 훈련세트의 클래스 비율 ##
        print(np.unique(y_train, return_counts = True))
        ## 훈련세트의 클래스 비율 ##
        print(np.unique(y_test, return_counts = True))
        
        #self.x_train, self.x_test, self.y_train, self.y_test = x_train, x_test, y_train, y_test
        return x_train, x_test, y_train, y_test
    
    def LSTM_attention_building(self, x_train, x_test, y_train, y_test):
        self.x_train, self.x_test, self.y_train, self.y_test = x_train, x_test, y_train, y_test
        # 2층-양방항 구조의 LSTM 모델을 생성한다.
        K.clear_session()     # 모델 생성전에 tensorflow의 graph 영역을 clear한다.

        xInput = Input(batch_shape=(None, x_train.shape[1], x_train.shape[2]))  #batch_shape=(환자 수,time step,feature 또는 컬럼 개수)


        x = LSTM(20, return_sequences=True)(xInput)
        #x = Bidirectional(LSTM(10))(x)

        # Bidirectional(LSTM(10,activation='relu',implementation=2,
        #                                    return_sequences=True,
        #                                    kernel_initializer='glorot_uniform'),
        #                          activity_regularizer=regularizers.l2(0.01))

        x = Attention(10)(x)
        x = Dropout(0.2)(x)
        x = Dense(1)(x)

        model = Model(xInput, x)


        #model.compile(loss='mse', optimizer='adam')  # mse 는 연속형라벨(ex>주가예측) 
        #model.compile(loss='categorical_crossentropy',optimizer='rmsprop', metrics=['accuracy']) # 2개 이상 라벨일 때 (ex>age group)
        # hinge NO

        #binary_crossentropy
        model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy']) # 1,0 binary 일 때 
        model.summary()
        self.model = model
        return model

    
    def class_balance_weight(self, output_domain_path, outcome_name, y_train):
        #클래스 불균형 하에서 모델은 1보다 0이 훨씬 더 많습니다. 또한 그렇게함으로써 훈련 손실을 최소화 할 수 있기 때문에 1보다 더 많은 0을 예측하는 방법을 배웁니다.
        #클래스 가중치를 사용하는 목적은 손실 함수를 변경하여 "쉬운 솔루션"(즉, 0 예측)으로 훈련 손실을 최소화 할 수 없도록하는 것이므로 1에 더 높은 가중치를 사용하는 것이 좋습니다.
        #가중치를 설정하는 방법은, 만약 A : B = 3 : 1이면 A에 1을, B에 3을 곱해주는 식으로 데이터 구성비의 역을 곱해주는 것이 합리적이다


        # 차원 변경 해줘야 함
        y_train2=y_train[:,0]


        weights = class_weight.compute_class_weight(class_weight = 'balanced',
                                                    classes = np.unique(y_train),
                                                    y = y_train2)
        weights = {i : weights[i] for i in range(2)}
        print(weights)

        with open('{}/{}_class_weight.txt'.format(output_domain_path,outcome_name),'w',encoding='UTF-8') as f:
            f.write(str(weights))
        
        self.class_weight = weights
            
        return weights
    
    def early_stopping_prediction(self):    
        # epoch 반복 하는데, patience =50 동안 loss 변화가 없다면 early stop.
        model = self.model
        x_train = self.x_train
        y_train = self.y_train
        x_test = self.x_test
        y_test = self.y_test
        class_weight = self.class_weight
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=100)
        
        # Early stopping 객체에 의해 training 이 중지되었을 때, 그 상태가 이전 모델에 비해 validation error 가 높은 상태 일 수 있음.
        # 따라서 가장 validation performance 가 좋은 모델을 저장하는 것이 필요한데, 이를 위해 ModelCheckPoint 사용
        mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', save_best_only=True)
        h=model.fit(x_train, y_train, epochs=50,  #############################################################################
                         batch_size=28, verbose=1, validation_split=0.1,
                         class_weight=class_weight,
                         callbacks=[es,mc])  # callbacks=[es,mc,visualize]
        y_hat = model.predict(x_test, batch_size=28)
        self.y_hat =  y_hat
        self.h = h
        return h, y_hat
    
    def classification_report(self, output_domain_path, outcome_name):
        y_hat = self.y_hat
        y_test = self.y_test        
        y_pred = (y_hat>=0.5).astype(int) # astype 없으면 true, false
        self.y_pred = y_pred
        
        res = classification_report(y_test, y_pred, target_names=['label 0', 'label 1'])
        print(res)

        with open('{}/{}_CLF_report.txt'.format(output_domain_path,outcome_name),'w',encoding='UTF-8') as f:
            f.write(str(res))
            
    def model_performance_evaluation(self, output_path, outcome_name):
        
        cf = confusion_matrix(self.y_test, self.y_pred, labels=[1,0])
        TP, FP, FN, TN = cf[0][0], cf[1][0], cf[0][1], cf[1][1]
        
        result = {}
        result['TP'] = TP
        result['FP'] = FP
        result['FN'] = FN
        result['TN'] = TN
        
        result['precision'] = TP/(TP+FP)
        result['specificity'] = TN/(TN+FP)
        result['sensitivity'] = TP/(TP+FN) 
        result['recall'] = result['sensitivity'] # recall = sensitivity
        result['accuracy'] = (TP+TN) / (FP+FN+TP+TN)
        result['f1score'] = 2*result['precision']*result['recall']/(result['precision']+result['recall'])
        
        out = open('{}/model_performance_evaluation.txt'.format(output_path),'a')
        out.write(str(outcome_name) + '///')
        out.write(str(TP) + '///')
        out.write(str(TN) + '///')
        out.write(str(FP) + '///')
        out.write(str(FN) + '///')
        out.write('{:.3}'.format(result['precision']) + '///')
        out.write('{:.3}'.format(result['specificity']) + '///')
        out.write('{:.3}'.format(result['accuracy']) + '///')
        out.write('{:.3}'.format(result['recall'])+ '///')
        out.write('{:.3}'.format(result['f1score']) )
        out.write('\n')        
        out.close()
        
        # print("TP", TP) 
        # print("TN", TN)
        # print("FP", FP)
        # print("FN", FN)
        # print("precision", '{:.3%}'.format(result['precision']))
        # print("specificity", '{:.3%}'.format(result['specificity']))
        # print("accuracy", '{:.3%}'.format(result['accuracy']))
        # print("recall", '{:.3%}'.format(result['recall']))
        # print("f1_score", '{:.3%}'.format(result['f1score']))    
        # from pycm import *
        # cm = ConfusionMatrix(actual_vector=np.array(c.y_test.ravel()), predict_vector=np.array(c.y_pred.ravel()))
        # cm.classes
        # cm.table
        # print(cm)
        return
            
    def confusion_matrix_figure(self, output_domain_path, outcome_name):        
        import matplotlib.pyplot as plt
        y_pred = self.y_pred
        y_test = self.y_test

        class_names=[0,1]
        
        # plot_confusion_matrix function

        def plot_confusion_matrix(y_test, y_pred, classes,
                                  normalize=False,
                                  title=None,
                                  cmap=plt.cm.Blues):
            """
            This function prints and plots the confusion matrix.
            Normalization can be applied by setting `normalize=True`.
            """
            if not title:
                if normalize:
                    title = 'Normalized confusion matrix'
                else:
                    title = 'Confusion matrix, without normalization'

            # Compute confusion matrix
            cm = confusion_matrix(y_test, y_pred)

            if normalize:
                cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                print("Normalized confusion matrix")
            else:
                print('Confusion matrix, without normalization')

            print(cm)

            fig, ax = plt.subplots()
            im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
            ax.figure.colorbar(im, ax=ax)
            # We want to show all ticks...
            ax.set(xticks=np.arange(cm.shape[1]),
                   yticks=np.arange(cm.shape[0]),
                   # ... and label them with the respective list entries
                   xticklabels=classes, yticklabels=classes,
                   title=title,
                   ylabel='True label',
                   xlabel='Predicted label')

            # Rotate the tick labels and set their alignment.
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                     rotation_mode="anchor")


            # Loop over data dimensions and create text annotations.
            fmt = '.2f' if normalize else 'd'
            thresh = cm.max() / 2.
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, format(cm[i, j], fmt),
                            ha="center", va="center",
                            color="white" if cm[i, j] > thresh else "black")
            fig.tight_layout()
            return ax



        plot_confusion_matrix(y_test, y_pred, classes=class_names, normalize=True)

        plt.savefig('{}/{}_CM.png'.format(output_domain_path,outcome_name), format='png',
                    dpi=300, facecolor='white', transparent=True,  bbox_inches='tight')
        return
    
    def confusion_matrix_figure2(self, output_domain_path, outcome_name):        
        import seaborn as sns
        import matplotlib.pyplot as plt
        y_pred = self.y_pred
        y_test = self.y_test
        
        plt.figure()
        cf = confusion_matrix(y_test, y_pred, labels=[1,0])
        cf_norm = confusion_matrix(y_test, y_pred, normalize='true', labels=[1,0])
        group_names = ['TP', 'FN', 'FP', 'TN']
        group_counts = ['{}'.format(value) for value in cf.flatten()]
        group_percentages = ['{:.2}'.format(value) for value in cf_norm.flatten()]
        labels = ['{}\n{}\n({})'.format(v1, v2, v3) for v1, v2, v3 in zip(group_names,group_counts,group_percentages)]
        labels = np.asarray(labels).reshape(2,2)
        cm_figure = sns.heatmap(cf_norm, annot=labels, fmt='', xticklabels=['1','0'], yticklabels=['1','0'], cmap='Blues')
        cm_figure.set_title('Confusion matrix')
        cm_figure.set_xlabel('Predicted label')
        cm_figure.set_ylabel('True label')
        plt.setp(cm_figure.get_yticklabels(), rotation=0)
        plt.savefig('{}/{}_CM2.png'.format(output_domain_path,outcome_name), format='png',
                    dpi=300, facecolor='white', transparent=True,  bbox_inches='tight')
        return
    
    def ROC_AUC(self, output_domain_path, outcome_name):
        
        # Compute ROC curve and ROC area for each class
        from sklearn.metrics import roc_curve, auc
        import matplotlib.pyplot as plt

        y_test = self.y_test
        y_hat = self.y_hat
        
        
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(1):
            fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_hat[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            AUC = round(roc_auc[i], 4)
        print('AUC:', AUC ) 
        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_hat.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        plt.figure()
        lw = 1
        plt.plot(fpr[0], tpr[0], color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[0])
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")

        plt.savefig('{}/{}_ROC.png'.format(output_domain_path,outcome_name) , format='png',
                    dpi=300, facecolor='white', transparent=True,  bbox_inches='tight')

        from sklearn.metrics import accuracy_score
        y_hat2 = np.where(y_hat>0.5, 1, 0)
        ACC = round(accuracy_score(y_hat2, y_test), 5)*100
        print("ACC:",ACC)
        
        # to draw mean ROC Curve
        np.savez('{}/{}_dat.npz'.format(output_domain_path, outcome_name), y_hat=y_hat, y_test=y_test)
        
        return AUC, ACC
        
    def loss(self, output_domain_path, outcome_name):
        # Loss history
        h = self.h
        
        y_vloss = h.history['val_loss']
        y_loss = h.history['loss']
        
        plt.figure(figsize=(8, 4))
        
        x_len = np.arange(len(y_loss))
        plt.plot(x_len, y_vloss, marker='.', c='red', label="Validation-set Loss")
        plt.plot(x_len, y_loss, marker='.', c='blue', label="Train-set Loss")

        plt.title("Loss History")
        plt.grid()

        plt.xlabel("epoch")
        plt.ylabel("loss")

        plt.savefig('{}/{}_loss.png'.format(output_domain_path,outcome_name), format='png',
                    dpi=300, facecolor='white', transparent=True,  bbox_inches='tight')
        return
    
    def attention_heatmap(self, new_col, output_domain_path, outcome_name):
        from keract import get_activations
        
        model = self.model
        x_train = self.x_train
        
        names = [layer.name for layer in model.layers]        
        activations = get_activations(model,x_train, layer_names='attention_weight')
        activations = [i for i in activations.values()]
        activations = np.array(activations) 

        attention_matrix = np.nanmean(activations, axis=0).squeeze()
        a=np.swapaxes(attention_matrix, 0, 1)
        
        features = new_col[2:]
        # features.insert(0,'age')
        # features.insert(1,'sex')
        print(features)        
        arranged_indices = [i for i in range(len(features))]
        A = a.T[arranged_indices] 
        A = np.sort(A)[::-1]
        
        A2 = np.max(A, axis=1)
        A2 = A2.reshape(A2.shape[0],1)

        f = np.array(features)
        f2 = f.reshape(f.shape[0],1)

        import_f = np.concatenate([A2,f2],axis=1)
        import_f2 = np.flip(import_f[import_f[:, 0].argsort()])
        pd.DataFrame(import_f2).to_csv('{}/{}_features_all.txt'.format(output_domain_path,outcome_name), index=False, header=False)
        
        selected_feature = import_f2[0:20]
        selected_feature = [f_ for [f_, v] in selected_feature]
        selected_feature_index = np.where(f2==selected_feature)[0]
        selected_A = A[selected_feature_index,:]
        selected_f = f[selected_feature_index]
        print(selected_A)
        print(selected_f)
        
        if_ = import_f2[0][0]
        print('importance feature:', if_)

        ### heatmap
        plt.figure(figsize = (26, 16))
        plt.imshow(selected_A, cmap = 'coolwarm')
        plt.xticks(range(selected_A.shape[1]))
        plt.yticks(range(selected_A.shape[0]), selected_f)

        # Loop over data dimensions and create text annotations.
        for y in range(selected_A.shape[0]):
           for x in range(selected_A.shape[1]):
                text = plt.text(x, y + 0.1, '%.4f' % selected_A[y, x], horizontalalignment='center', verticalalignment='center',)

        plt.colorbar()
        plt.title('Importance features', fontsize=20)
        plt.xlabel('Day', fontsize=15)
        plt.ylabel('Features', fontsize=15)

        plt.savefig('{}/{}_heatmap.png'.format(output_domain_path,outcome_name), format='png',
                    dpi=300, facecolor='white', transparent=True,  bbox_inches='tight')
        
        return if_


    def k_fold_cross_validation(self, x, y, output_domain_path, outcome_name):
        from sklearn.metrics import roc_curve, auc
        from sklearn.metrics import accuracy_score
        accuracy =[]
        skf = StratifiedKFold(n_splits =5, random_state=None)  ############## 5-fold
        skf.get_n_splits(x, y)

        # X is the feature set and y is the target(=label)
        for train_index, test_index in skf.split(x, y):
            #print("Train:",train_index,'\n','\n', "Validation:",test_index)
            x1_train, x1_test = x[train_index], x[test_index]
            y1_train, y1_test = y[train_index], y[test_index]

            y_train2 = y1_train[:,0]
            weights = class_weight.compute_class_weight(class_weight = 'balanced',
                                                    classes = np.unique(y1_train),
                                                    y = y_train2)
            weights = {i : weights[i] for i in range(2)}
            #print(weights)


            K.clear_session() 
            xInput = Input(batch_shape=(None, x1_train.shape[1], x1_train.shape[2]))
            X = LSTM(20, return_sequences=True)(xInput)
            Bidirectional(LSTM(10,activation='relu',implementation=2,
                                               return_sequences=True,
                                               kernel_initializer='glorot_uniform'),
                                     activity_regularizer=regularizers.l2(0.01))
            X = Attention(10)(X)
            X = Dropout(0.2)(X)
            X = Dense(1)(X)
            model = Model(xInput, X)
            model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy']) # 1,0 binary 일 때 


            es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
            mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', save_best_only=True)
            h=model.fit(x1_train, y1_train, epochs=50,  
                         batch_size=28, class_weight=weights, verbose=1, validation_split=0.1,   
                         callbacks=[es,mc])  # callbacks=[es,mc,visualize]    
            y_hat = model.predict(x1_test, batch_size=28)    

            y_hat2 = np.where(y_hat>0.5, 1, 0)

            #score = model.evaluate(y_hat, y1_test, batch_size=28)
            score = accuracy_score(y_hat2, y1_test)
            accuracy.append(score)

        print(accuracy)

        with open('{}/{}_k_fold_validation_ACC.txt'.format(output_domain_path,outcome_name),'w',encoding='UTF-8') as f:
            f.write(str(accuracy))
            f.close()
        
        return accuracy
    
    def plotRiskChangeOverTime(output_dir, outcome_name, nSamples=15):
        
        model = self.model
        X_test = self.x_test
        y_test = self.y_test
        
        for label in set(y_test.flatten()):
            
            y_indices = np.where(y_test.flatten() == label)[0]
            y_label_max_count = len(y_indices)
            nSamples = nSamples if y_label_max_count > nSamples else y_label_max_count
            y_indices = y_indices[:nSamples]
            X_test_indices = X_test[y_indices]

            names = [layer.name for layer in model.layers]
            activations = get_activations(model, X_test_indices, layer_names='attention_weight')
            activations = [i for i in activations.values()]
            activations = np.array(activations)
            attention_matrix = np.nanmean(activations, axis=0).squeeze()
            a=np.swapaxes(attention_matrix, 0, 1)

            arranged_indices = [i for i in range((nSamples))]
            A = a.T[arranged_indices] 
            plt.figure(figsize = (8, 12))
            plt.imshow(A, cmap = 'coolwarm')
            plt.xticks(range(A.shape[1]))
            plt.yticks(range(A.shape[0]),["patient {}".format(i+1) for i in range(nSamples)])
            # y_scores = model.predict(X_test_indices)
            # y_scores = (y_scores>=0.5).astype(int) 
            # plt.yticks(range(A.shape[0]),["patient {} {}".format(i+1, y_scores[i]) for i in range(nSamples)])

            # Loop over data dimensions and create text annotations.
            for y in range(A.shape[0]):
                for x in range(A.shape[1]):
                    text = plt.text(x, y + 0.1, '%.4f' % A[y, x], horizontalalignment='center', verticalalignment='center',)

            status = "abnormal" if label == 1 else "normal"
            plt.colorbar()
            plt.title('Risk change over time ({})'.format(status), fontsize=20)
            plt.xlabel('Day', fontsize=15)
            plt.ylabel('Patients', fontsize=15)
            plt.savefig('{}/{}_heatmap_{}.png'.format(output_dir, outcome_name, status), format='png',
                dpi=300, facecolor='white', transparent=True,  bbox_inches='tight')
