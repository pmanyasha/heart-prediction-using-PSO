from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index1.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.form['message']
    bot_response = get_chatbot_response(user_message)
    return bot_response

def get_chatbot_response(message):
    # Your chatbot logic goes here
    # This is just a simple example
    if message == 'hi':
        return 'Hello! How can I assist you?'
    elif message == 'bye':
        return 'Goodbye! Have a great day!'
    else:
        return "I'm sorry, but I didn't understand that."

if __name__ == '__main__':
    app.run(debug=True)