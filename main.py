from flask import Flask, render_template, request, make_response
import pyttsx3
from time import sleep
from threading import Thread
from myTraining import pn
import pickle

app = Flask(__name__)

file = open('model.pkl', 'rb')
model = pickle.load(file)
file.close()
chatbot_responses = {
    'greeting': "Hello! How can I assist you today?",
    'consultation': "I recommend consulting a cardiologist for a comprehensive assessment of your heart health. Regular monitoring is crucial, especially with a family history of heart disease or other risk factors. Embrace a healthy lifestyle, incorporating regular exercise, a balanced diet, and stress management to maintain optimal heart health.",
    'goodbye': "Thank you for using our service. Take care!"
}

def intro_voice():
    converter = pyttsx3.init()
    converter.setProperty('rate', 180)
    converter.setProperty('volume', 1.0)
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[1].id)
    converter.say("Hello!")
    converter.say("""Welcome to university of zimbabwe,Here is my Heart Disease Prediction platform.""")
    sleep(1)
    converter.runAndWait()

def predicted_voice(text1, text22):
    text1 = text1
    text22 = text22
    converter = pyttsx3.init()
    converter.setProperty('rate', 180)
    converter.setProperty('volume', 1.0)
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[1].id)
    converter.say(text1)
    converter.say(text22)
    sleep(1)
    converter.runAndWait()

@app.route('/', methods=['GET', 'POST'])
def hello_world():
    if request.method == 'POST':
        myDict = request.form
        name = str(myDict['name'])
        age = int(myDict['age'])
        anaemia = int(myDict['anaemia'])
        creatinine_phosphokinase = int(myDict['creatinine_phosphokinase'])
        diabetes = int(myDict['diabetes'])
        ejection_fraction = int(myDict['ejection_fraction'])
        high_blood_pressure = int(myDict['high_blood_pressure'])
        platelets = int(myDict['platelets'])
        serum_creatinine = int(myDict['serum_creatinine'])
        serum_sodium = int(myDict['serum_sodium'])
        sex = int(myDict['sex'])
        smoking = int(myDict['smoking'])
        maxhr = int(myDict['maxhr'])
        cp = int(myDict['cp'])
        cholesterol = int(myDict['cholesterol'])
        fh = int(myDict['fh'])
        Medication = int(myDict['Medication'])
        inputFeatures = [age, anaemia, creatinine_phosphokinase, diabetes, ejection_fraction, high_blood_pressure,
                         platelets, serum_creatinine, serum_sodium, sex, smoking, cp, cholesterol, maxhr]
        infProb = model.predict_proba([inputFeatures])[0][1]
        pv = round(infProb * 100)
        consultation_response = chatbot_responses['goodbye']
        if pv > 70:
            consultation_response = chatbot_responses['consultation']

    # Return prediction result and consultation response
        return render_template('result.html', prediction=pv, consultation_response=consultation_response)
        predicted = pn(fh, Medication)
        print(infProb)
        text1 = "Hey" + " " + name
        text2 = "You have a probability of Heart Disease of"
        text22 = "You have a probability of Heart Disease of {} percentage".format(predicted)
        text3 = "%"
        print(predicted)
        print(pv)

        thr = Thread(target=predicted_voice, args=[text1, text22])
        thr.start()


    thr = Thread(target=intro_voice)
    thr.start()
# Virtual consultation


    return make_response(render_template('index.html'))

if __name__ == "__main__":
    app.run(debug=True)