
from app import app

if __name__=='__main__':
  app.run(debug=True, port=4000) # lo del port puede ser un problema... a no ser que lo montemos sobre un container
  


