import pandas as pd

from flask import Flask, render_template, request, send_file
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


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/train', methods=['POST'])
def train():
    classifier_name = request.form.get('classifier')
    parameters = get_parameters(request.form)

    # Carregue seu conjunto de dados aqui
    dataset_path = 'static/data/Most_Visited_Destination_in_2018_and_2019.csv'
    dataset = pd.read_csv(dataset_path)

    # Supondo que as features estão em colunas específicas, ajuste conforme necessário
    X = dataset.iloc[:, :-1]  # Todas as colunas, exceto a última (assumindo que a última é a coluna de rótulos)
    y = dataset.iloc[:, -1]  # A última coluna é a coluna de rótulos

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
    classes = dataset['target_column'].unique()  # Substitua 'target_column' pelo nome da coluna de rótulos
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, classes)

    # Converta a imagem para uma string base64
    image_str = plot_to_base64()

    return render_template('result.html', accuracy=accuracy, precision=precision, recall=recall, f1=f1, image=image_str)
def get_parameters(form_data):
    # Implemente a lógica para extrair os parâmetros do formulário
    classifier_name = form_data.get('classifier')
    param1 = form_data.get('param1')
    param2 = form_data.get('param2')

    # Retorne os parâmetros como um dicionário
    return {'param1': param1, 'param2': param2}

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
    plt.savefig('static/confusion_matrix.png')

def plot_to_base64():
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    img_str = base64.b64encode(img.read()).decode()
    return img_str


@app.route('/download_confusion_matrix')
def download_confusion_matrix():
    return send_file('static/confusion_matrix.png', as_attachment=True)