
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI(title="General Knowledge Chatbot API")

# Load the saved model
try:
    with open('chatbot_model.pkl', 'rb') as f:
        model_data = pickle.load(f)
    vectorizer = model_data['vectorizer']
    questions = model_data['questions']
    answers = model_data['answers']
    questions_vector = model_data['questions_vector']
except Exception as e:
    raise RuntimeError(f"Failed to load chatbot model: {e}")

class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    question: str
    matched_question: str
    answer: str
    confidence: float

@app.post("/ask", response_model=AnswerResponse)
def ask_chatbot(request: QuestionRequest):
    user_question = request.question.strip()
    if not user_question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    # Transform and check similarity
    user_vector = vectorizer.transform([user_question])
    score = cosine_similarity(user_vector, questions_vector).flatten()
    best_index = score.argmax()
    best_score = float(score[best_index])

    if best_score < 0.2:
        return {
            "question": user_question,
            "matched_question": "None",
            "answer": "Sorry, I don't have a relevant answer.",
            "confidence": best_score
        }

    return {
        "question": user_question,
        "matched_question": questions[best_index],
        "answer": answers[best_index],
        "confidence": best_score
    }

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
