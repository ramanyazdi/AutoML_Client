from celery import Celery
import requests

app = Celery('tasks', broker='amqp://guest:guest@localhost:5672/', backend='db+sqlite:///db.async')

@app.task
def upload_to_training_server(filename):
    # upload a file to the training server
    
    f = open(filename, "rb")
    files = {'dataset': f}
    r = requests.post('http://127.0.0.1:5000/upload', files=files)
    f.close()
    return 'response from training server: ' + str(r.text)

@app.task
def train_model(payload):
    res = requests.post('http://127.0.0.1:3000/train_model', json=payload)

    return 'response from training server: ' + str(res.text)