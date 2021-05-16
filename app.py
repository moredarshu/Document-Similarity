import joblib
import numpy as np
from flask import Flask,request,render_template
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
docsim_vec = joblib.load('DocSimVec.pkl')


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/similarity',methods=['POST'])
def similarity():
    output = ' '
    input_doc = [x for x in request.form.values()]
    input_vec1 = docsim_vec.transform(input_doc)
    
    output=cosine_similarity(input_vec1.toarray()[0].reshape(1,-1),input_vec1.toarray()[1].reshape(1,-1))
    if output == 0:
         prediction = 'not similar'
    else:
         prediction = 'similar'   

    return render_template('index.html',prediction_text='The Documents are :{}'.format(prediction))

if __name__ == "__main__":
    app.run(debug=True)