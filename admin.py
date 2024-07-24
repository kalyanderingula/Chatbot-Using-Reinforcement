from flask import Flask, request, jsonify, render_template
from model import ReinforcementChatBot, load_unknown_questions, save_unknown_questions

app = Flask(__name__)
chatbot = ReinforcementChatBot()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get', methods=['GET'])
def get_bot_response():
    user_text = request.args.get('msg')
    response = chatbot.ask_question(user_text)
    return jsonify(response)

@app.route('/update', methods=['POST'])
def update_bot_response():
    data = request.get_json()
    question = data['question']
    answer = data['answer']
    chatbot.update_q_table(question, answer)
    unknown_questions = load_unknown_questions()
    unknown_questions = [q for q in unknown_questions if q != question]
    save_unknown_questions(unknown_questions)
    return jsonify(success=True)

@app.route('/admin', methods=['GET'])
def admin_interface():
    unknown_questions = load_unknown_questions()
    return render_template('admin.html', questions=unknown_questions)

if __name__ == "__main__":
    app.run(debug=True)
