#from keras.models import Model
#from keras.layers import Input, Dense, LSTM, Bidirectional
#from keras import backend as K
#from keras import backend as K

import sys
from sklearn.utils import class_weight
from keract import get_activations
from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import *
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import classification_report

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# pd.set_option('display.max_colwidth', -1)  #각 컬럼 width 최대로 
# pd.set_option('display.max_rows', 50)      # display 50개 까지 

#data_dir = '/home/taehyun/Project/cardio/data/'

def get_clf_eval(y_test, y_pred=None, pred_proba=None):
    from sklearn.metrics import confusion_matrix, roc_auc_score, matthews_corrcoef
        
    cf = confusion_matrix(y_test, y_pred, labels=[1,0])
    roc_auc = roc_auc_score(y_test, pred_proba)
    mcc = matthews_corrcoef(y_test, y_pred)
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
    result['roc_auc'] = roc_auc
    result['mcc'] = mcc
    
    print("TP", result['TP'])
    print("FP", result['FP'])
    print("FN", result['FN'])
    print("TN", result['TN'])
    print("precision", '{:.3%}'.format(result['precision']))
    print("specificity", '{:.3%}'.format(result['specificity']))
    print("sensitivity", '{:.3%}'.format(result['sensitivity']))
    print("recall", '{:.3%}'.format(result['recall']))
    print("accuracy", '{:.3%}'.format(result['accuracy']))
    print("f1score", '{:.3%}'.format(result['f1score']))
    print("roc_auc", '{:.3%}'.format(result['roc_auc']))
    print("mcc", '{:.3%}'.format(result['mcc']))

def class_balance_weight(output_domain_path, outcome_name, y_train):
    #클래스 불균형 하에서 모델은 1보다 0이 훨씬 더 많습니다. 또한 그렇게함으로써 훈련 손실을 최소화 할 수 있기 때문에 1보다 더 많은 0을 예측하는 방법을 배웁니다.
    #클래스 가중치를 사용하는 목적은 손실 함수를 변경하여 "쉬운 솔루션"(즉, 0 예측)으로 훈련 손실을 최소화 할 수 없도록하는 것이므로 1에 더 높은 가중치를 사용하는 것이 좋습니다.
    #가중치를 설정하는 방법은, 만약 A : B = 3 : 1이면 A에 1을, B에 3을 곱해주는 식으로 데이터 구성비의 역을 곱해주는 것이 합리적이다
    # 차원 변경 해줘야 함
    # y_train2=y_train[:,0]
    weights = class_weight.compute_class_weight(class_weight = 'balanced',
                                                classes = np.unique(y_train),
                                                y = y_train)
    weights = {i : weights[i] for i in range(2)}
    print(weights)
    with open('{}/{}_class_weight.txt'.format(output_domain_path,outcome_name),'w',encoding='UTF-8') as f:
        f.write(str(weights))
        
    return weights

def make_plot_tree(xgb_model, output_domain_path, outcome_name, rankdir=None):
    import xgboost as xgb
    xgb.plot_tree(xgb_model, num_trees=0, rankdir=rankdir)
    fig = plt.gcf()
    fig.set_size_inches(15, 10)
    plt.savefig('{}/{}_tree.png'.format(output_domain_path, outcome_name), dpi=300, facecolor='white', transparent=True,  bbox_inches='tight')
    plt.show()

    
def make_plot_importance(xgb_model, output_domain_path, outcome_name):
    import xgboost as xgb
    xgb.plot_importance(xgb_model)
    fig = plt.gcf()
    fig.set_size_inches(15, 10)
    plt.savefig('{}/{}_plot_importance.png'.format(output_domain_path, outcome_name), dpi=300, facecolor='white', transparent=True,  bbox_inches='tight')
    plt.show()
    
def save_xgb_model_json(xgb_model, output_domain_path, outcome_name):
    import json
    with open("{}/{}_model.json".format(output_domain_path, outcome_name), "w") as outfile:
        json.dump(json.loads(xgb_model.save_config()), outfile)
    config = json.loads(xgb_model.save_config())
    xgb_model.save_model("{}/{}_model2.json".format(output_domain_path, outcome_name))
    print(config)
    
def clf_report(y_test, y_pred, output_domain_path, outcome_name):
    res = classification_report(y_test, y_pred, target_names=['label 0', 'label 1'])
    print(res)
    with open('{}/{}_CLF_report.txt'.format(output_domain_path,outcome_name),'w',encoding='UTF-8') as f:
        f.write(str(res))
        
def model_performance_evaluation(y_test, y_pred, pred_proba, output_path, outcome_name):
    from sklearn.metrics import confusion_matrix, roc_auc_score, precision_recall_curve, auc


    lr_precision, lr_recall, _  = precision_recall_curve(y_test, y_pred)
    pr_auc = auc(lr_recall, lr_precision)
    cf = confusion_matrix(y_test, y_pred, labels=[1,0])
    roc_auc = roc_auc_score(y_test, pred_proba)
    mcc = matthews_corrcoef(y_test, y_pred)
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
    result['roc_auc'] = roc_auc
    result['mcc'] = mcc
    result['pr_auc'] = pr_auc
    
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
    out.write('{:.3}'.format(result['f1score'])+ '///')
    out.write('{:.3}'.format(result['roc_auc'])+ '///')
    out.write('{:.3}'.format(result['mcc'])+ '///')
    out.write('{:.3}'.format(result['pr_auc']))
    out.write('\n')        
    out.close()
    
    print("TP", TP) 
    print("TN", TN)
    print("FP", FP)
    print("FN", FN)
    print("precision", '{:.3%}'.format(result['precision']))
    print("specificity", '{:.3%}'.format(result['specificity']))
    print("accuracy", '{:.3%}'.format(result['accuracy']))
    print("recall", '{:.3%}'.format(result['recall']))
    print("f1_score", '{:.3%}'.format(result['f1score']))
    print("roc_auc", '{:.3%}'.format(result['roc_auc']))
    print("mcc", '{:.3%}'.format(result['mcc']))    
    print("pr_auc", '{:.3%}'.format(result['pr_auc']))
    # from pycm import *
    # cm = ConfusionMatrix(actual_vector=np.array(c.y_test.ravel()), predict_vector=np.array(c.y_pred.ravel()))
    # cm.classes
    # cm.table
    # print(cm)
    return
        
def confusion_matrix_figure(y_test, y_pred, output_domain_path, outcome_name):        
    import matplotlib.pyplot as plt
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

def confusion_matrix_figure2(y_test, y_pred, output_domain_path, outcome_name):        
    from sklearn.metrics import confusion_matrix
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
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

def ROC_AUC(y_pred, y_test, output_domain_path, outcome_name):
    
    # Compute ROC curve and ROC area for each class
    from sklearn.metrics import roc_curve, auc
    import matplotlib.pyplot as plt
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)
    AUC = round(roc_auc, 4)
    
    # for i in range(1):
    #     fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred[:, i])
    #     roc_auc[i] = auc(fpr[i], tpr[i])
    #     AUC = round(roc_auc[i], 4)
    print('AUC:', AUC ) 
    # Compute micro-average ROC curve and ROC area
    # fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_pred.ravel())
    # roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    plt.figure()
    lw = 1
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC(Receiver operating characteristic) curve')
    plt.legend(loc="lower right")

    plt.savefig('{}/{}_ROC.png'.format(output_domain_path,outcome_name) , format='png',
                dpi=300, facecolor='white', transparent=True,  bbox_inches='tight')

    from sklearn.metrics import accuracy_score
    y_pred2 = np.where(y_pred>0.5, 1, 0)
    ACC = round(accuracy_score(y_pred2, y_test), 5)*100
    print("ACC:",ACC)
    
    # to draw mean ROC Curve
    np.savez('{}/{}_dat.npz'.format(output_domain_path, outcome_name), y_hat=y_pred, y_test=y_test)
    
    return AUC, ACC

def PR_AUC(y_pred, y_test, output_domain_path, outcome_name):
    
    # Compute ROC curve and ROC area for each class
    from sklearn.metrics import roc_curve, auc, precision_recall_curve
    import matplotlib.pyplot as plt
    
    lr_precision, lr_recall, _  = precision_recall_curve(y_test, y_pred)
    roc_auc = auc(lr_recall, lr_precision)
    AUC = round(roc_auc, 4)
    
    # for i in range(1):
    #     fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred[:, i])
    #     roc_auc[i] = auc(fpr[i], tpr[i])
    #     AUC = round(roc_auc[i], 4)
    print('AUC:', AUC ) 
    # Compute micro-average ROC curve and ROC area
    # fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_pred.ravel())
    # roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    plt.figure()
    lw = 1
    plt.plot(lr_recall, lr_precision, color='darkorange',
             lw=lw, label='PR curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower right")

    plt.savefig('{}/{}_PRAUC.png'.format(output_domain_path,outcome_name) , format='png',
                dpi=300, facecolor='white', transparent=True,  bbox_inches='tight')

    from sklearn.metrics import accuracy_score
    y_pred2 = np.where(y_pred>0.5, 1, 0)
    ACC = round(accuracy_score(y_pred2, y_test), 5)*100
    print("ACC:",ACC)
    
    # to draw mean ROC Curve
    np.savez('{}/{}_pr_dat.npz'.format(output_domain_path, outcome_name), y_hat=y_pred, y_test=y_test)
    
    return AUC, ACC
