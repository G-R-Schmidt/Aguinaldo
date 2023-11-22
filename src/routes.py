import os

import pandas as pd

from flask import Flask, render_template, request, send_file, current_app
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from io import BytesIO
import base64

from src import app
from src.forms import MLForm


@app.route('/')
def index():
    form = MLForm()  # Crie uma instância do formulário
    return render_template('index.html', form=form)

@app.route('/train', methods=['POST'])
def train():
    classifier_name = request.form.get('classifier')
    parameters = get_parameters(request.form, classifier_name)

    # Carregue o conjunto de dados Iris
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    # Divida o conjunto de dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Inicialize o classificador selecionado com os parâmetros escolhidos
    classifier = get_classifier(classifier_name, parameters)

    # Treine o classificador
    classifier.fit(X_train, y_train)

    # Faça previsões
    y_pred = classifier.predict(X_test)

    # Calcule as métricas
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')

    # Crie a matriz de confusão
    classes = iris.target_names.tolist()
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, classes)

    # Converta a imagem para uma string base64
    image_str = plot_to_base64()

    result = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'image': image_str,
        'confusion_matrix_path': 'static/conf_photos/confusion_matrix.png'
    }

    return render_template('index.html', form=MLForm(), result=result)
def get_parameters(form_data, classifier_name):
    # Implemente a lógica para extrair os parâmetros do formulário
    params = {}
    for i in range(1, 4):  # Assumindo que você tem até 3 parâmetros, ajuste conforme necessário
        param_key = f'param{i}'
        param_value = form_data.get(param_key)
        params[param_key] = param_value

    # Lógica específica para cada classificador
    if classifier_name == 'knn':
        # Adicione lógica específica para KNN
        params['n_neighbors'] = int(params.get('param1'))
        params['weights'] = params.get('param2')

    elif classifier_name == 'svm':
        # Adicione lógica específica para SVM
        params['C'] = float(params.get('param1'))
        params['kernel'] = params.get('param2')

    elif classifier_name == 'mlp':
        # Adicione lógica específica para MLP
        params['hidden_layer_size'] = int(params.get('param1'))
        params['max_iter'] = int(params.get('param2'))

    elif classifier_name == 'dt':
        # Adicione lógica específica para Decision Tree
        params['max_depth'] = int(params.get('param1'))

    elif classifier_name == 'rf':
        # Adicione lógica específica para Random Forest
        params['n_estimators'] = int(params.get('param1'))

    return params

def get_classifier(name, params):
    # Implemente a lógica para inicializar o classificador com os parâmetros
    if name == 'knn':
        classifier = KNeighborsClassifier(n_neighbors=int(params['param1']))
    elif name == 'svm':
        classifier = SVC(C=float(params['param1']), kernel=params['param2'])
    elif name == 'mlp':
        classifier = MLPClassifier(hidden_layer_sizes=(int(params['param1']),), max_iter=int(params['param2']))
    elif name == 'dt':
        classifier = DecisionTreeClassifier(max_depth=int(params['param1']))
    elif name == 'rf':
        classifier = RandomForestClassifier(n_estimators=int(params['param1']))

    return classifier


def plot_confusion_matrix(cm, classes):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    # plt.savefig('static/conf_photos/confusion_matrix.png')

    # Verifique se o diretório 'static/conf_photos' existe, senão crie
    conf_photos_dir = 'static/conf_photos'
    if not os.path.exists(conf_photos_dir):
        os.makedirs(conf_photos_dir)

    # Salve a matriz de confusão no diretório 'static/conf_photos' com um nome específico
    plt.savefig(os.path.join(conf_photos_dir, 'confusion_matrix.png'))

def plot_to_base64():
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    img_str = base64.b64encode(img.read()).decode()
    return img_str


@app.route('/download_confusion_matrix')
def download_confusion_matrix():
    # return send_file('static/conf_photos/confusion_matrix.png', as_attachment=True, mimetype='image/png')
    conf_photos_dir = os.path.join(current_app.root_path, 'static', 'conf_photos')
    file_path = os.path.join(conf_photos_dir, 'confusion_matrix.png')
    return send_file(file_path, as_attachment=True, mimetype='image/png')