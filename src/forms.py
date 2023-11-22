from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, SelectField

class FormularioML(FlaskForm):
    classificador = SelectField('Classificador', choices=[('knn', 'KNN'), ('svm', 'SVM'), ('mlp', 'MLP'), ('dt', 'Decision Tree'), ('rf', 'Random Forest')])
    parametro1 = StringField('Parâmetro 1')
    parametro2 = StringField('Parâmetro 2')
    parametro3 = StringField('Parâmetro 3')
    enviar = SubmitField('Treinar')