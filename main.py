from fastapi import FastAPI
from pydantic import BaseModel

from model import load_model, model_predict

model, tokenizer = load_model()
app = FastAPI()


class Ticket(BaseModel):
    message: str


@app.post("/ticket_support_classification")
def classify_ticket(ticket: Ticket):
    assert isinstance(ticket.message, str)
    return {
        "ticket_message": ticket.message,
        "ticket_category": int(model_predict(model, tokenizer, [ticket.message])),
    }
