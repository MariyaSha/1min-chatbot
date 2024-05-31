FROM python:3

RUN pip install --user -U nltk
RUN pip install torch
RUN pip install transformers

COPY question_answering.py ./

ENTRYPOINT python3 question_answering.py