FROM python:3.7
WORKDIR /support_ticket_classification
ADD ./requirements.txt ./requirements.txt
RUN pip install -r requirements.txt
ADD . .
CMD [ "uvicorn", "--host", "0.0.0.0", "main:app"]