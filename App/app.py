from flask import Flask, render_template, request
import pipelines as pipe

app = Flask(__name__)


@app.route("/")
def Home():
    #print("HOME IN")
    return render_template("index.html")

@app.route("/index.html")
def Back():
    return render_template("index.html")

@app.route('/predict', methods = ['post'])
def predict():
    #print("PREDICT IN")
    output = [x for x in request.form.values()]
    #print("OUTPUT: ", output)
    #print("OUTPUT LENGTH: ", len(output))

    preds = pipe.predict_loan_output(output)
    preds[0] = "Yes" if int(preds[0]) == 0 else "No"
    preds[1] = [float(l) for l in preds[1][0]]
    IA = ((preds[1][2]/100)) * (preds[1][0] - ((preds[1][2]/100)*preds[1][0]))
    preds.append((IA / (preds[1][0])) * 100) # Interest Rate
    preds.append(int(preds[1][0] / preds[1][1])) # Loan Tenure
    #print("RESULTS : ", preds)

    return render_template("submit.html", n = [preds[0], f"{float(preds[1][1]):.2f}",\
         f"{float(preds[1][0]):.2f}", f"{float(preds[1][2]):.2f}%", f"{int(preds[3])} Months"])

if __name__ == "__main__":
    app.run()
  