from flask import Flask, render_template, request
from DSAI_Utility import *

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("DSAI_Index.html")

@app.route('/get')
def chatbot_response():
    msg = request.args.get('msg')
    bag = preprocess(msg, words)
    ints = predict_class(bag, classes, model)
    res = getResponse(ints, intents)
    return res

if __name__ == "__main__":
    model, intents, words, classes = load_dependencies()
    app.run(debug=True)