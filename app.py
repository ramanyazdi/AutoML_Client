from flask import Flask, render_template, request, session, url_for, redirect, jsonify
from werkzeug.utils import secure_filename
from flask_sqlalchemy import SQLAlchemy
import pandas as pd
import io
import eda_draw
import os
import time
import requests
import json
from celery import Celery
import asynchronous
import numpy as np
from imblearn.over_sampling import SMOTE

app = Flask(__name__)
app.secret_key = "super secret key"

# database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///data.db'
db = SQLAlchemy(app)

# eda plots save folder
EDA_FOLDER = os.path.join('static', 'eda')
app.config['UPLOAD_FOLDER'] = EDA_FOLDER


class Dataset(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(200), nullable=False)
    classification = db.Column(db.Integer, default=0)

# write json
def write_json(dictionary, filename='instance/database.json'):
    with open(filename, "w") as outfile:
        json.dump(dictionary, outfile)
    
    print("successfully wrote", dictionary, "to", filename)

# read json
def read_json(filename='instance/database.json'):
    with open(filename, 'r') as openfile:
        dictionary = json.load(openfile)
    
    return dictionary

# update json
def update_json(key, value, filename='instance/database.json'):
    dictionary = read_json(filename)
    dictionary[key] = value
    write_json(dictionary, filename)
    print("successfully updated dictionary")

# filter out columns
def filtercolumn(datasetname,columns, label):
    # reading from pandas
    df = pd.read_csv(datasetname)
    df = df[columns + [label]]
    return df

# get the value of something in json
def get_json(key, filename='instance/database.json'):
    dictionary = read_json(filename)
    return dictionary[key]


@app.route('/')  
def main():  
    return render_template("index.html")


@app.route('/results_csv', methods=['GET']) # call linux to create results.csv files
def create_results_csv():
    r = requests.get(url = 'http://localhost:5000/results_csv')

    return jsonify( r.json() )
    

@app.route('/download_models', methods=['POST'])
def download_models():
    if request.method == 'POST':
        downloaded = []

        for model_name in request.files.keys():
            model = request.files[model_name]
            model.save('downloads/'+model.filename)
            downloaded.append[model_name]
            print(model_name, "saved")

        return jsonify({'downloaded':downloaded})


@app.route('/full_trained_models')
def full_trained_models():
    """request linux server to send fully trained models"""
    r = requests.get('http://127.0.0.1:5000/upload_models')
    
    return 'response from training server: ' + str(r.text)


@app.route('/upload', methods = ['GET', 'POST'])
def upload_file():
    save_dir = 'D:/Dataset2/'

    if request.method == 'POST':
        f = request.files['file']
        # filename = save_dir + secure_filename(f.filename)
        filename = save_dir + f.filename
        f.save(filename)

        # asynchronous task
        asynchronous.upload_to_training_server.delay(filename)

        # add dataset to db
        database = {}
        database['dataset_filename'] = filename
        write_json(database)
        return redirect(url_for('.display_data'))

        # try:
        #     db.session.add(new_dataset)
        #     db.session.commit()
        #     return redirect(url_for('.display_data'))
        # except:
        #     return "There was an issue adding dataset to db"

@app.route('/eda', methods = ['GET', 'POST'])
def eda():
    if request.method == 'POST':
        columns=request.form.getlist('columns')
        label=request.form['label']

        # add label to db
        update_json('label',label)

        # prevent label from appearing twice
        if label in columns:
            columns.remove(label)


        # save columns into json file
        update_json("features", columns)

        # get dataset from db
        # current_dataset = Dataset.query.order_by(Dataset.id.desc()).first()
        current_dataset = read_json()
        dataset_filename = current_dataset['dataset_filename']

        df = filtercolumn(dataset_filename, columns,label)

        # reading from pandas
        # df = pd.read_csv(dataset_filename)

        # unique classes for label
        classes = list(df[label].unique()).sort()
        update_json('classes',classes)

        binary_classification = False
        if df[label].nunique() == 2:
            binary_classification = True
        
        update_json('binary_classification',binary_classification)

        # filter out columns
        # df = df[columns + [label]]
        describe = df.describe().applymap('{:,.2f}'.format)
        operation = list(describe.index)
        describe.insert(loc=0, column='', value=operation)

        # df to use for feature selection
        # df_feature = df_feature[columns]
        feature_dict = {"Features":columns}
        df_feature = pd.DataFrame.from_dict(feature_dict)


        # clear eda folder first
        for file in os.listdir('static/eda'):
            os.remove('static/eda/'+file)

        # draw and save the graphs
        eda_draw.draw_all(df, label)

        # distplot filenames
        distplots = [i for i in os.listdir('static/eda') if 'distplot' in i]
        temp = []
        for plot in distplots: # add full filepath
            temp.append(os.path.join(app.config['UPLOAD_FOLDER'], plot))
        distplots = temp

        return render_template("eda.html", 
                                columns=request.form.getlist('columns'),
                                label=request.form['label'],

                                # df.describe
                                column_names=describe.columns.values, 
                                row_data=list(describe.values.tolist()), 
                                zip=zip,
                                link_column='',
                                
                                correlation_plot=os.path.join(app.config['UPLOAD_FOLDER'], 'correlation.png'),

                                distplots = distplots,

                                # metadata_df
                                df_feature_column_names=df_feature.columns.values, 
                                df_feature_row_data=list(df_feature.values.tolist()), 

                                )

@app.route('/permutation', methods=['GET'])
def permutation():
    r = requests.get(url = 'http://127.0.0.1:5000/permutation')

    permutations = r.json()
    
    json_file = read_json()
    features = json_file['features']

    for key in permutations:
        value_list, label_list = eda_draw.get_sorted_permutation(permutations[key], features)
        eda_draw.draw_permutation_importance(value_list, label_list, f"{key}.png")

    print('\n')
    print("permutation importance", permutations)
    
    return jsonify(permutations)

@app.route('/predictions', methods=['GET'])
def draw_predictions():
    r = requests.get(url = 'http://127.0.0.1:5000/predictions')
    predictions_dict = r.json()

    binary_classification = get_json('binary_classification')
    dataframes_list = eda_draw.generate_insights(predictions_dict, binary_classification, get_json('classes'))

    return jsonify({'dataframes_list':dataframes_list,'binary_classification':binary_classification})


@app.route('/train', methods = ['GET', 'POST'])
def train():
    if request.method == 'POST':
        columns=request.form.getlist('columns')
        time_left_for_this_task=request.form['time_left_for_this_task']
        per_run_time_limit=request.form['per_run_time_limit']
        test_set_size = request.form['test_set_size']
        cv = request.form['cv_folds']

        # sending post to the training server
        payload = {}

        # get dataset from db
        # dataset_filename = Dataset.query.order_by(Dataset.id.desc()).first().filename.split('/')[-1]
        current_dataset = read_json()
        dataset_filename = current_dataset['dataset_filename'].split('/')[-1]
        payload['dataset_filename'] = dataset_filename

        payload['columns'] = columns
        payload['time_left_for_this_task'] = time_left_for_this_task
        payload['per_run_time_limit'] = per_run_time_limit
        payload['test_set_size'] = test_set_size

        # label = Dataset.query.order_by(Dataset.id.desc()).first().label
        current_dataset = read_json()
        label = current_dataset['label']

        return render_template('train.html',
                                columns=json.dumps(columns),
                                time_left_for_this_task=time_left_for_this_task,
                                per_run_time_limit=per_run_time_limit,
                                dataset_filename=dataset_filename,
                                label=label,
                                test_set_size=test_set_size,
                                cv=cv,
                                # train_job=train_job
                                payload=payload
        )

@app.route('/documentation')
def documentation():
    return render_template('documentation.html')

@app.route('/train_model', methods=['GET'])
def train_model():
    if request.method == 'GET':
        print(request.args)

        columns=request.args['columns']
        time_left_for_this_task=request.args['time_left_for_this_task']
        per_run_time_limit=request.args['per_run_time_limit']
        test_set_size = request.args['test_set_size']

        payload = {}

        # get dataset from db
        # dataset_filename = Dataset.query.order_by(Dataset.id.desc()).first().filename.split('/')[-1]
        current_dataset = read_json()
        dataset_filename = current_dataset['dataset_filename'].split('/')[-1]

        payload['dataset_filename'] = dataset_filename
        payload['test_set_size'] = test_set_size
        payload['columns'] = columns
        payload['time_left_for_this_task'] = time_left_for_this_task
        payload['per_run_time_limit'] = per_run_time_limit

        # return payload
        
        print("payload from windows", payload)
        res = requests.post('http://127.0.0.1:5000/train_model', params=request.args)
        return jsonify(res.text)

@app.route('/display_data')
def display_data():
    # get dataset from db
    # current_dataset = Dataset.query.order_by(Dataset.id.desc()).first()
    current_dataset = read_json()
    dataset_filename = current_dataset['dataset_filename']

    # reading from pandas
    # df = pd.read_csv(current_dataset.filename)
    df = pd.read_csv(dataset_filename)
    

    # metadata df
    buf = io.StringIO()
    df.info(buf=buf)
    s = buf.getvalue()
    lines = [line.split() for line in s.splitlines()[3:-2]]
    metadata_df = pd.DataFrame(lines[2:], columns=lines[0])

    # prevent displaying entire dataset
    df = df.head()
    return render_template("display_data.html", 
                            column_names=df.columns.values, 
                            row_data=list(df.values.tolist()), 
                            zip=zip,
                            # filename=current_dataset.filename.split('/')[-1],
                            filename=dataset_filename.split('/')[-1],
                            
                            # metadata_df
                            metadata_column_names=metadata_df.columns.values, 
                            metadata_row_data=list(metadata_df.values.tolist()), 
                            )

@app.route('/augmentation', methods=['POST'])
def augmentation(): 
    #dataset from eda page (user determine lable and deselect unrelated columns)
    columns=request.form.getlist('columns') # get columns
    label=request.form['targetcolumn'] #get labels
    update_json('label',label) # add label to db  
    if label in columns: # prevent label from appearing twice
        columns.remove(label)
    current_dataset = Dataset.query.order_by(Dataset.id.desc()).first()# get dataset from db
    current_dataset = read_json()
    dataset_filename = current_dataset['dataset_filename']
    
    df = filtercolumn(dataset_filename, columns, label) # final dataset ready for improvment 
    
    


    # get the desired number of rows for augmentation from user
    # desired_rows = 500
    desired_rows = request.form['desired_row']
    desired_rows_int = int(desired_rows)
    # Data augmentation 
    augmented_data = df.sample(n=desired_rows_int - len(df), replace=True) # Duplicate the existing rows randomly to increase the number of rows
    augmented_df = pd.concat([df, augmented_data], axis=0) # Concatenate the original dataset with the augmented rows
    augmented_df = augmented_df.sample(frac=1).reset_index(drop=True) # Shuffle the rows to introduce diversity
    
    save_dir = 'datasets/'
    dataset_filename = current_dataset['dataset_filename'].split('/')[-1]
    new_dataset_filename = dataset_filename.replace('.csv',"")+"_augmented.csv"
    new_ds_fullpath = save_dir+new_dataset_filename
    augmented_df.to_csv(new_ds_fullpath, index=False)
    save_dir='D:/Dataset2/'
    new_ds_fullpath = save_dir+new_dataset_filename
    augmented_df.to_csv(new_ds_fullpath, index=False)

    update_json('dataset_filename',new_ds_fullpath)

    # show the result on improve.html 
    print("Original dataset shape:", df.shape)
    print("Augmented dataset shape:", augmented_df.shape)
    
    test_set_size= request.form['test_set_size']
    columns=request.form.getlist('columns')
    time_left_for_this_task=request.form['time_left_for_this_task']
    per_run_time_limit=request.form['per_run_time_limit']
    cv_folds=request.form['cv_folds']
    class_distribution = df[label].value_counts()
    return render_template('improve.html',
                            augmented= augmented_df.shape,
                            class_distribution=class_distribution,
                            test_set_size=test_set_size,
                            columns=columns,
                            time_left_for_this_task=time_left_for_this_task,
                            per_run_time_limit=per_run_time_limit,
                            cv_folds=cv_folds
                           )

@app.route('/smote', methods=['POST'])
def smote(): 
    #dataset from eda page (user determine lable and deselect unrelated columns)
    columns=request.form.getlist('columns') # get columns
    label=request.form['targetcolumn'] #get labels
    update_json('label',label) # add label to db  
    if label in columns: # prevent label from appearing twice
        columns.remove(label)
    current_dataset = Dataset.query.order_by(Dataset.id.desc()).first()# get dataset from db
    current_dataset = read_json()
    dataset_filename = current_dataset['dataset_filename']
    
    df = filtercolumn(dataset_filename, columns, label) # final dataset ready for improvment 
    

    df = df.fillna(df.mean())

    
    # Data Improvment 
    # Assuming your features are stored in 'X' and the target variable in 'y'
    X = df.drop(label, axis=1)
    y = df[label]

    # get the desired proportion of the minority class for oversampling
    # desired_proportion = 3 
    desired_proportion = request.form['desired_proportion']
    desired_proportion_int = int(desired_proportion)
    # Calculate the number of samples needed for the minority class
    min_class_samples = int(desired_proportion_int * len(y[y == 1]))
    # Create an instance of SMOTE with the desired sampling strategy
    smote = SMOTE(sampling_strategy={0: len(y[y == 0]), 1: min_class_samples}, random_state=42)
    # Perform oversampling using SMOTE
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # Check the new sample size and class distribution
    total_samples = len(X_resampled)
    class_distribution = y_resampled.value_counts()

    print("Total samples after oversampling:", total_samples)
    print("Class distribution after oversampling:")
    print(class_distribution)

    # Create a new DataFrame with the oversampled data
    smote_df = pd.concat([X_resampled, y_resampled], axis=1)

    save_dir = 'datasets/'
    dataset_filename = current_dataset['dataset_filename'].split('/')[-1]
    new_dataset_filename = dataset_filename.replace('.csv',"")+"_smote.csv"
    new_ds_fullpath = save_dir+new_dataset_filename
    smote_df.to_csv(new_ds_fullpath, index=False)
    save_dir='D:/Dataset2/'
    new_ds_fullpath = save_dir+new_dataset_filename
    smote_df.to_csv(new_ds_fullpath, index=False)

    update_json('dataset_filename',new_ds_fullpath)

    test_set_size= request.form['test_set_size']
    columns=request.form.getlist('columns')
    time_left_for_this_task=request.form['time_left_for_this_task']
    per_run_time_limit=request.form['per_run_time_limit']
    cv_folds=request.form['cv_folds']

    return render_template("improve.html",
                            totalsample= total_samples,
                            class_distribution= class_distribution,
                            augmented= smote_df.shape,
                            test_set_size=test_set_size,
                            columns=columns,
                            time_left_for_this_task=time_left_for_this_task,
                            per_run_time_limit=per_run_time_limit,
                            cv_folds=cv_folds
                            )

@app.route('/train_after_improve', methods=['POST'])
def train_after_improve():
   
    columns=eval(request.form.get('columns'))
    time_left_for_this_task=request.form['time_left_for_this_task']
    per_run_time_limit=request.form['per_run_time_limit']
    test_set_size = request.form['test_set_size']
    cv = request.form['cv_folds']

    # sending post to the training server
    payload = {}

    # get dataset from db
    # dataset_filename = Dataset.query.order_by(Dataset.id.desc()).first().filename.split('/')[-1]
    current_dataset = read_json()
    dataset_filename = current_dataset['dataset_filename'].split('/')[-1]
    payload['dataset_filename'] = dataset_filename

    payload['columns'] = columns
    payload['time_left_for_this_task'] = time_left_for_this_task
    payload['per_run_time_limit'] = per_run_time_limit
    payload['test_set_size'] = test_set_size

    # label = Dataset.query.order_by(Dataset.id.desc()).first().label
    # current_dataset = read_json()
    label = current_dataset['label']

    return render_template('train.html',
                            columns=json.dumps(columns),
                            time_left_for_this_task=time_left_for_this_task,
                            per_run_time_limit=per_run_time_limit,
                            dataset_filename=dataset_filename,
                            label=label,
                            test_set_size=test_set_size,
                            cv=cv,
                            # train_job=train_job
                            payload=payload
)

if __name__ == '__main__':
    app.run(debug = True, port=3000)