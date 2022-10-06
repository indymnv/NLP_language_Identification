from flask import Flask
import os

app=Flask(__name__,template_folder = 'templates', static_folder='static')
app.config['SECRET_KEY']='batman&robin'

# puede que no sea necesario
#APP_ROOT=os.path.dirname(os.path.abspath(__file__))

from app import routes