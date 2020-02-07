from flask import Flask, render_template, url_for
import sys

app = Flask(__name__)
app = Flask(__name__, template_folder='template') 

@app.route('/')
def index():
    return render_template('index.html')

def prarg():
    if len(sys.argv) >= 2:
        name = sys.argv[1]
    else:
        name = 'Default'
    print("ENtered" + name) 

if __name__ == "__main__":
    app.run(debug=True)  