from flask import Flask
from flask_sqlalchemy import SQLAlchemy

# Inicializa a aplicação Flask
app = Flask(__name__, static_url_path='/static')
app.static_folder = 'static'

# Configurações do banco de dados SQLite
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///projeto_novo.db'
app.config['SECRET_KEY'] = 'chave_secreta_123'
app.config['UPLOAD_FOLDER'] = 'static/fotos_posts'

# Inicializa a extensão SQLAlchemy
banco_de_dados = SQLAlchemy(app)

# Importa as rotas do módulo src
from src import rotas
