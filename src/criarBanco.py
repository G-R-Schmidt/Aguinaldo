from src import app, banco_de_dados

# Cria as tabelas no banco de dados
with app.app_context():
    banco_de_dados.create_all()
