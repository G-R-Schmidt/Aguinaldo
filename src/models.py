from src import banco_de_dados

class ModeloML(banco_de_dados.Model):
    id = banco_de_dados.Column(banco_de_dados.Integer, primary_key=True)
    nome = banco_de_dados.Column(banco_de_dados.String(100), nullable=False)
    parametros = banco_de_dados.Column(banco_de_dados.String(100), nullable=False)
    acuracia = banco_de_dados.Column(banco_de_dados.Float, nullable=False)
    precisao = banco_de_dados.Column(banco_de_dados.Float, nullable=False)
    recall = banco_de_dados.Column(banco_de_dados.Float, nullable=False)
    f1 = banco_de_dados.Column(banco_de_dados.Float, nullable=False)
    data_criacao = banco_de_dados.Column(banco_de_dados.DateTime, default=banco_de_dados.func.current_timestamp())

    def __repr__(self):
        return f"ModeloML('{self.nome}', '{self.parametros}', '{self.data_criacao}')"
