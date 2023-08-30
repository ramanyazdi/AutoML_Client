import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn import preprocessing

def generate_insights(predictions_dict, binary_classification, labels):
    """generate insights about the model performance"""
    y_cv = predictions_dict['y_cv']

    dataframes_list = {}

    # get the model names
    for key in predictions_dict['predictions']:
        predict = predictions_dict['predictions_int_dict'][key]
        dataframes_list[key] = generate_clf_report(y_cv, predict).to_dict()

        draw_confusion_matrix(y_cv, predict, labels, save_name=f"{key}_confusion_matrix.png", save_path='static/model_results')


        # draw roc curve only for binary classification
        if binary_classification:
            predict_proba = np.array(predictions_dict['predictions'][key])[:, 1]
            draw_roc_curve(y_cv, predict_proba, save_name=f"{key}_roc.png", save_path='static/model_results')

    return dataframes_list




def generate_clf_report(y_cv, predictions):
    report = classification_report(y_cv, predictions, output_dict=True)
    clf_df = pd.DataFrame(report).transpose()
    clf_df.drop('accuracy', inplace=True)
    
    return clf_df


def draw_confusion_matrix(y_cv, predictions, labels, save_name='confusion_matrix.png', save_path='static/model_results'):
    cm = confusion_matrix(y_cv, predictions, labels=labels)
    
    # Creating a dataframe for a array-formatted Confusion matrix,so it will be easy for plotting.
    cm_df = pd.DataFrame(cm,
                         index = labels, 
                         columns = labels)

    #Plotting the confusion matrix
    plt.figure(figsize=(5,4))
    sns.heatmap(cm_df, annot=True, fmt='.4g', cmap="Greens")
    plt.title('Confusion Matrix')
    plt.ylabel('Actal Values')
    plt.xlabel('Predicted Values')

    plt.savefig(f"{save_path}/{save_name}", bbox_inches='tight')


def draw_roc_curve(y_cv, predictions, save_name='roc.png', save_path='static/model_results'):
    fpr, tpr, thresholds = metrics.roc_curve(y_cv, predictions)
    roc_auc = metrics.auc(fpr, tpr)
    display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,
                                     estimator_name='')
    display.plot()
    plt.title('ROC Curve')
    
    plt.savefig(f"{save_path}/{save_name}", bbox_inches='tight')




# correlation heatmap
def correlation_heatmap(df, path):
    correlation = df.corr().round(2)
    plt.figure(figsize = (14,7))
    correlation_plot = sns.heatmap(correlation, annot = True, cmap = 'YlOrBr')

    # saving figure
    fig = correlation_plot.get_figure()
    fig.savefig(f"{path}/correlation.png", bbox_inches='tight')

# for continuous variables
def dist_plot(df, column, path, index=0):
    plt.figure(figsize = (5,4))
    sns.set(color_codes = True)

    fig = sns.distplot(df[column], kde = False).get_figure()
    fig.savefig(f"{path}/distplot_{index}_{column}.png", bbox_inches='tight')
    


def draw_label_frequency(df, label, save_path):
    # plot labels
    count = df[label].value_counts()
    plt.bar(count.index.to_numpy().astype('str'), count.to_numpy(), color ='maroon')
    plt.xlabel("Labels")
    plt.ylabel("Frequency")
    plt.grid(b=True)
    plt.savefig(f"{save_path}/label_frequency.png", bbox_inches='tight')


def prepare_data_helper(df):
    columns = list(df)
    
    for c, dtype in zip(columns, list(df.dtypes)):
        if dtype == 'object':
            le = preprocessing.LabelEncoder()
            le.fit(df[c])
            df[c] = le.transform(df[c])
    
    return df


def draw_all(df, label):
    """
    makes the call to all the functions to generate and save plots
    """
    df = prepare_data_helper(df)
    print(df)

    path = "static/eda" # savepath of the plots
    columns = list(df)

    correlation_heatmap(df, path)
    

    for index, col in enumerate(columns):
        dist_plot(df, col, path, index=index)
    
    
    draw_label_frequency(df, label, path)


    


def get_sorted_permutation(result, data_columns):
    """
    for drawing permutation feature importance graphs
    """
    value_list = []
    label_list = []
    
    # sort the result.importances_mean
    sorted_result = sorted(zip(result, data_columns), reverse=True)
    
    for item in sorted_result:
        value_list.append(item[0])
        label_list.append(item[1])
        
    return value_list, label_list


def draw_permutation_importance(value_list, label_list, save_name='file.png', save_path='static/feature_importance'):
    plt.figure(figsize = (14,7))
    
    # if feature name is too long, slant the feature name
    if len(max(label_list, key=len)) > 13:
        plt.xticks(rotation=45)

    plt.ylabel('Mean feature importance')
    plt.bar(label_list, value_list)
    plt.grid(b=True)
    
    plt.savefig(f"{save_path}/{save_name}", bbox_inches='tight')