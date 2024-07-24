import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import json
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = []
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0   # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = models.Sequential()
        model.add(layers.Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(layers.Dense(24, activation='relu'))
        model.add(layers.Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

class ReinforcementChatBot:
    def __init__(self):
        self.load_data()
        self.state_size = len(self.questions)
        self.action_size = len(self.answers)
        self.agent = DQNAgent(self.state_size, self.action_size)
        self.batch_size = 32
        self.load_memory()
        self.vectorizer = TfidfVectorizer()
        self.vectorizer.fit(self.questions)  # Fit on initial questions

    def load_data(self):
        try:
            with open('data.json', 'r') as file:
                data = json.load(file)
            self.questions = data['questions']
            self.answers = data['answers']
        except FileNotFoundError:
            self.questions = []
            self.answers = []

    def save_data(self):
        with open('data.json', 'w') as file:
            data = {
                "questions": self.questions,
                "answers": self.answers
            }
            json.dump(data, file)

    def load_memory(self):
        try:
            with open('chatbot_memory.json', 'r') as file:
                self.memory = json.load(file)
        except FileNotFoundError:
            self.memory = []

    def save_memory(self):
        with open('chatbot_memory.json', 'w') as file:
            json.dump(self.memory, file)

    def ask_question(self, question):
        # Use NLP to find the most similar question
        question_vector = self.vectorizer.transform([question])
        tfidf_matrix = self.vectorizer.transform(self.questions)
        similarities = cosine_similarity(question_vector, tfidf_matrix)
        most_similar = np.argmax(similarities)
        
        if similarities[0, most_similar] > 0.5:  # Similarity threshold
            return self.answers[most_similar]
        else:
            self.log_question(question)
            return "Sorry, I don't know the answer to that. I've notified the admin."

    def log_question(self, question):
        unknown_questions = load_unknown_questions()
        if question not in unknown_questions:
            with open('unknown_questions.txt', 'a') as file:
                file.write(question + '\n')

    def update_q_table(self, question, answer):
        if question not in self.questions:
            self.questions.append(question)
            self.answers.append(answer)
            self.state_size = len(self.questions)
            self.action_size = len(self.answers)
            self.vectorizer = TfidfVectorizer()
            self.vectorizer.fit(self.questions)  # Re-fit on updated questions
            self.agent = DQNAgent(self.state_size, self.action_size)
        self.save_data()
        self.save_memory()

def load_unknown_questions():
    try:
        with open('unknown_questions.txt', 'r') as file:
            questions = file.readlines()
        return [q.strip() for q in questions]
    except FileNotFoundError:
        return []

def save_unknown_questions(questions):
    with open('unknown_questions.txt', 'w') as file:
        for question in questions:
            file.write(question + '\n')
