from flask import Flask, jsonify, render_template, request, redirect
import subprocess as sp
import ai21
from toxic import resultAPI




def toxicity_comment(comment):
    result= resultAPI(comment)

    return result

def paraphrase_comment(comment):

    ai21.api_key = ('x62I8ZV78hONc5QWuOsRUw3wOJCDsIzm')

    response = ai21.Paraphrase.execute(text=comment, max_returned_sequences=1)
    result = ''
    for suggestion in response['suggestions']:
        # suggestion = response['suggestions'][1]['text']
        result = result + suggestion['text']+'\n'

    return result




app = Flask(__name__, template_folder="templates/project3")
@app.route("/")
def homepge():
    return index()
    # return scan()

@app.route("/index")
def index():
    return render_template("index.html")

@app.route("/aboutUs")
def aboutUs():
    return render_template("aboutUs.html")

@app.route("/case_study")
def case_study():
    return render_template("case_study.html")

@app.route("/our_work")
def our_work():
    return render_template("our_work.html")

@app.route("/get_in_touch")
def get_in_touch():
    return render_template("get_in_touch.html")

@app.route("/scan",methods = ['POST', 'GET'])
def scan():
    toxic_comment=''
    if request.method == 'POST':  
        toxic_comment = request.form['comment']
    if (toxic_comment):
        toxic_result = toxicity_comment(toxic_comment)
        return jsonify({'output':toxic_result})
    return render_template("scan.html")

@app.route("/paraphrase",methods = ['POST', 'GET'])
def paraphrase():
    para_comment=''
    if request.method == 'POST':  
        para_comment = request.form['comment']
    if (para_comment):
        para_result = paraphrase_comment(para_comment)
        return jsonify({'output':para_result})
    # return jsonify({'error' : 'Error!'})
    return render_template("paraphrase.html")

@app.route("/acc/sudeep_linkedin")
def sudeep_linkedin():
    return redirect("https://www.linkedin.com/in/sudeep-roy-54a7b6247/")

@app.route("/acc/sudeep_github")
def sudeep_github():
    return redirect("https://github.com/Sudeeproy520")

#--------------------------------------------------------------------------------
@app.route("/acc/raunak_linkedin")
def raunak_linkedin():
    return redirect("https://www.linkedin.com/in/raunak-palewar-0bb833190/")
@app.route("/acc/raunak_github")
def raunak_github():
    return redirect("https://github.com/raunakpalewar")
#--------------------------------------------------------------------------------
@app.route("/acc/mandeep_linkedin")
def mandeep_linkedin():
    return redirect("https://www.linkedin.com/in/mandeep-sharma-83647a247")
@app.route("/acc/mandeep_github")
def mandeep_github():
    return redirect("https://github.com/mandeepsharma19")
#--------------------------------------------------------------------------------
@app.route("/acc/sameer_linkedin")
def sameer_linkedin():
    return redirect("")
@app.route("/acc/sameer_github")
def sameer_github():
    return redirect("")
#--------------------------------------------------------------------------------
@app.route("/acc/rohan_linkedin")
def rohan_linkedin():
    return redirect("https://www.linkedin.com/in/rohan-malhotra-405502233")
@app.route("/acc/rohan_github")
def rohan_github():
    return redirect("https://github.com/RohanMalhotra297")
        
    




@app.errorhandler(404)
def page_not_found(e):
    return render_template('error404.html'), 404


if __name__ == '__main__':  
   app.run()

