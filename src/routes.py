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
from src.forms import FormularioML

@app.route('/')
def inicio():
    formulario = FormularioML()  # Cria uma instância do formulário
    return render_template('index.html', formulario=formulario)

@app.route('/treinar', methods=['POST'])
def treinar():
    nome_classificador = request.form.get('classificador')
    parametros = obter_parametros(request.form, nome_classificador)

    # Carrega o conjunto de dados Iris
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    # Divide o conjunto de dados em treino e teste
    X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, random_state=42)

    # Inicializa o classificador selecionado com os parâmetros escolhidos
    classificador = obter_classificador(nome_classificador, parametros)

    # Treina o classificador
    classificador.fit(X_treino, y_treino)

    # Faz previsões
    y_predito = classificador.predict(X_teste)

    # Calcula as métricas
    acuracia = accuracy_score(y_teste, y_predito)
    precisao = precision_score(y_teste, y_predito, average='macro')
    recall = recall_score(y_teste, y_predito, average='macro')
    f1 = f1_score(y_teste, y_predito, average='macro')

    # Cria a matriz de confusão
    classes = iris.target_names.tolist()
    matriz_confusao = confusion_matrix(y_teste, y_predito)
    plotar_matriz_confusao(matriz_confusao, classes)

    # Converte a imagem para uma string base64
    imagem_str = plotar_para_base64()

    resultado = {
        'acuracia': acuracia,
        'precisao': precisao,
        'recall': recall,
        'f1': f1,
        'imagem': imagem_str,
        'caminho_matriz_confusao': 'static/conf_photos/matriz_confusao.png'
    }

    return render_template('index.html', formulario=FormularioML(), resultado=resultado)

def obter_parametros(dados_formulario, nome_classificador):
    # Implementa a lógica para extrair os parâmetros do formulário
    params = {}
    for i in range(1, 4):  # Assumindo que você tem até 3 parâmetros, ajuste conforme necessário
        chave_param = f'parametro{i}'
        valor_param = dados_formulario.get(chave_param)
        params[chave_param] = valor_param

    # Lógica específica para cada classificador
    if nome_classificador == 'knn':
        # Adiciona lógica específica para KNN
        params['n_vizinhos'] = int(params.get('parametro1'))
        params['pesos'] = params.get('parametro2')

    elif nome_classificador == 'svm':
        # Adiciona lógica específica para SVM
        params['C'] = float(params.get('parametro1'))
        params['kernel'] = params.get('parametro2')

    elif nome_classificador == 'mlp':
        # Adiciona lógica específica para MLP
        params['tamanho_camada_oculta'] = int(params.get('parametro1'))
        params['max_iteracoes'] = int(params.get('parametro2'))

    elif nome_classificador == 'dt':
        # Adiciona lógica específica para Decision Tree
        params['max_profundidade'] = int(params.get('parametro1'))

    elif nome_classificador == 'rf':
        # Adiciona lógica específica para Random Forest
        params['n_estimadores'] = int(params.get('parametro1'))

    return params

def obter_classificador(nome, params):
    # Implementa a lógica para inicializar o classificador com os parâmetros
    if nome == 'knn':
        classificador = KNeighborsClassifier(n_neighbors=int(params['parametro1']))
    elif nome == 'svm':
        classificador = SVC(C=float(params['parametro1']), kernel=params['parametro2'])
    elif nome == 'mlp':
        classificador = MLPClassifier(hidden_layer_sizes=(int(params['parametro1']),), max_iter=int(params['parametro2']))
    elif nome == 'dt':
        classificador = DecisionTreeClassifier(max_depth=int(params['parametro1']))
    elif nome == 'rf':
        classificador = RandomForestClassifier(n_estimators=int(params['parametro1']))

    return classificador

def plotar_matriz_confusao(matriz_confusao, classes):
    plt.figure(figsize=(8, 6))
    sns.heatmap(matriz_confusao, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Previsto')
    plt.ylabel('Real')
    plt.title('Matriz de Confusão')
    # plt.savefig('static/conf_photos/matriz_confusao.png')

    # Verifica se o diretório 'static/conf_photos' existe, senão cria
    diretorio_conf_photos = 'static/conf_photos'
    if not os.path.exists(diretorio_conf_photos):
        os.makedirs(diretorio_conf_photos)

    # Salva a matriz de confusão no diretório 'static/conf_photos' com um nome específico
    plt.savefig(os.path.join(diretorio_conf_photos, 'matriz_confusao.png'))

def plotar_para_base64():
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    img_str = base64.b64encode(img.read()).decode()
    return img_str

@app.route('/download_matriz_confusao')
def download_matriz_confusao():
    # return send_file('static/conf_photos/matriz_confusao.png', as_attachment=True, mimetype='image/png')
    diretorio_conf_photos = os.path.join(current_app.root_path, 'static', 'conf_photos')
    caminho_arquivo = os.path.join(diretorio_conf_photos, 'matriz_confusao.png')
    return send_file(caminho_arquivo, as_attachment=True, mimetype='image/png')