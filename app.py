import pandas as pd
from flask import Flask, request, render_template
import pickle
import warnings
warnings.filterwarnings('ignore')

app = Flask("__name__")

q = ""

@app.route("/")
def loadPage():
	return render_template('home.html', query="")


@app.route("/", methods=['POST'])
def predict():
    
    '''
    gender
    seniorcitizen 
    partner
    dependents
    tenure
    phoneservice
    multiplelines
    internetservice
    onlinesecurity
    onlinebackup
    deviceprotection 
    techsupport
    contract
    paperlessbilling
    paymentmethod
    monthlycharges
    '''
    

    
    inputQuery1 = request.form['query1']
    inputQuery2 = request.form['query2']
    inputQuery3 = request.form['query3']
    inputQuery4 = request.form['query4']
    inputQuery5 = request.form['query5']
    inputQuery6 = request.form['query6']
    inputQuery7 = request.form['query7']
    inputQuery8 = request.form['query8']
    inputQuery9 = request.form['query9']
    inputQuery10 = request.form['query10']
    inputQuery11 = request.form['query11']
    inputQuery12 = request.form['query12']
    inputQuery13 = request.form['query13']
    inputQuery14 = request.form['query14']
    inputQuery15 = request.form['query15']
    inputQuery16 = request.form['query16']

    # model = pickle.load(open("D:/Projects/churn_project/models/model_recall.pkl", "rb"))
    model = pickle.load(open("model_recall.pkl", "rb"))
    
    data = [[inputQuery1, inputQuery2, inputQuery3, inputQuery4, inputQuery5, inputQuery6, inputQuery7, 
             inputQuery8, inputQuery9, inputQuery10, inputQuery11, inputQuery12, inputQuery13, inputQuery14,
             inputQuery15, inputQuery16]]
    
    new_data = pd.DataFrame(data, columns = ['seniorcitizen', 'monthlycharges', 'gender', 
                                           'partner', 'dependents', 'phoneservice', 'multiplelines', 'internetservice',
                                           'onlinesecurity', 'onlinebackup', 'deviceprotection', 'techsupport',
                                           'contract', 'paperlessbilling',
                                           'paymentmethod', 'tenure'])
    
   
    single = model.predict(new_data)
    probablity = model.predict_proba(new_data)[:, 1]
    
    if single == 1:
        o1 = "This customer is likely to be churned!!"
        o2 = "Confidence: {}".format(probablity*100)
    else:
        o1 = "This customer is likely to continue!!"
        o2 = "Confidence: {}".format(probablity*100)
        
    return render_template('home.html', output1 = o1, output2 = o2)
                        #    query1 = request.form['query1'], 
                        #    query2 = request.form['query2'],
                        #    query3 = request.form['query3'],
                        #    query4 = request.form['query4'],
                        #    query5 = request.form['query5'], 
                        #    query6 = request.form['query6'], 
                        #    query7 = request.form['query7'], 
                        #    query8 = request.form['query8'], 
                        #    query9 = request.form['query9'], 
                        #    query10 = request.form['query10'], 
                        #    query11 = request.form['query11'], 
                        #    query12 = request.form['query12'], 
                        #    query13 = request.form['query13'], 
                        #    query14 = request.form['query14'], 
                        #    query15 = request.form['query15'], 
                        #    query16 = request.form['query16'])

if __name__ == "__main__":  
    app.run(host='0.0.0.0',port=5000)