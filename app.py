from flask import Flask, render_template, request, make_response
import pyttsx3
from threading import Thread
import pickle
from myTraining import pn

app = Flask(__name__)

# Load the trained model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Chatbot responses
chatbot_responses = {
    'greeting': "Hello! How can I assist you today?",
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
    converter.say("Welcome to the University of Zimbabwe. Here is my Heart Disease Prediction platform.")
    converter.runAndWait()

def predicted_voice(text1, text22):
    converter = pyttsx3.init()
    converter.setProperty('rate', 180)
    converter.setProperty('volume', 1.0)
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[1].id)
    converter.say(text1)
    converter.say(text22)
    converter.runAndWait()

def predict_heart_disease(**kwargs):
    try:
        input_features = [int(kwargs.get('age', 0)), int(kwargs.get('anaemia', 0)), 
                          int(kwargs.get('creatinine_phosphokinase', 0)), int(kwargs.get('diabetes', 0)),
                          int(kwargs.get('ejection_fraction', 0)), int(kwargs.get('high_blood_pressure', 0)),
                          int(kwargs.get('platelets', 0)), float(kwargs.get('serum_creatinine', 0)),
                          int(kwargs.get('serum_sodium', 0)), int(kwargs.get('sex', 0)),
                          int(kwargs.get('smoking', 0)), int(kwargs.get('cp', 0)),
                          int(kwargs.get('cholesterol', 0)), int(kwargs.get('maxhr', 0))]
        inf_prob = model.predict_proba([input_features])[0][1]
        return round(inf_prob * 100)
    except Exception as e:
        print("Error in prediction:", e)
        return None

@app.route('/', methods=['GET', 'POST'])
def hello_world():
    if request.method == 'POST':
        myDict = request.form.to_dict()
        name = myDict.get('name', '')
        
        # Predict heart disease probability
        prediction = predict_heart_disease(**myDict)
        if prediction is None:
            consultation_response = "Sorry, an error occurred while predicting. Please try again later."
        else:
            # Determine consultation response based on predicted percentage
            if prediction < 50:
                consultation_response = ("Based on the prediction, it is advisable to consult a medical professional for a thorough evaluation."
                                          "The prediction indicates a lower likelihood of heart-related issues, but it's still recommended to maintain a healthy lifestyle and regular check-ups."
                                          "While the prediction suggests a lower risk of heart problems, it's important to focus on preventive measures such as maintaining a balanced diet and engaging in regular exercise.")
            elif 50 <= prediction <= 70:
                consultation_response = ("The prediction indicates a moderate likelihood of heart-related concerns. It is recommended to follow up with a healthcare provider for further evaluation."
                                          "Based on the prediction, it's advisable to make positive changes to your lifestyle, such as adopting a heart-healthy diet and engaging in regular physical activity."
                                          "The prediction suggests a moderate risk of heart issues. Consider scheduling an appointment with a healthcare professional to discuss preventive measures and potential treatment options.")
            else:
                consultation_response = ("Based on the prediction, it is highly recommended to seek immediate medical attention and consultation with a healthcare professional."
                                          "The prediction indicates a higher likelihood of heart-related concerns. It's crucial to prioritize your health and consult with a doctor for a comprehensive evaluation and guidance."
                                          "Given the high predicted percentage, it is strongly advised to take prompt action by consulting a healthcare provider. Early intervention can significantly improve outcomes.")

            # Trigger voice prediction
            thr = Thread(target=predicted_voice, args=[f"Hey {name}", f"You have a probability of Heart Disease of {prediction}%"])
            thr.start()
            
            return render_template('result1.html', prediction=prediction, consultation_response=consultation_response)

    # Trigger welcome message
    thr = Thread(target=intro_voice)
    thr.start()

    return make_response(render_template('index.html'))

if __name__ == "__main__":
    app.run(debug=True)
