from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField

class MLForm(FlaskForm):
    classifier = StringField('Classificador')
    param1 = StringField('Parâmetro 1')
    param2 = StringField('Parâmetro 2')
    submit = SubmitField('Treinar')