from flask import jsonify
from flask import render_template

from app import app #, podria importar APP si lo llego a ocupar en app.py

@app.route('/',methods=['GET'])
def index():
  return render_template('index.html', titlle='Home')

@app.route('/home',methods=['GET'])
def home():
  return render_template('index.html', titlle='Home')

@app.route('/about')
def about():
  return render_template('about.html', titlle='Home')

# mas links pages
