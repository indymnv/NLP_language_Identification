from flask import jsonify
from flask import render_template
from app import process

from app import app #, podria importar APP si lo llego a ocupar en app.py

@app.route('/',methods=['GET'])
def home():
  return render_template('index.html', titlle='Home')

@app.route('/identification', methods=['GET'])
def runModel():
  valores =process.init()
  return render_template('index.html', output = valores)
