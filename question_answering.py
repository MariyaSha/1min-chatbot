from transformers import pipeline
import nltk.corpus
nltk.download('genesis')

# load text
context = nltk.corpus.genesis.raw(
    "english-kjv.txt"
)

# load question answering model
model = pipeline(
    model="deepset/roberta-base-squad2"
)

# receive user input continuously 
while True:
    # collect input
    user_input = input(
        "Your question:"
    )

    if user_input == "exit":
        break
    else:
        # enter input into model
        output = model(
            question=user_input, 
            context=context
        )

        print(
            # print likelihood 
            round(
                output['score'], 2
            ),
            # print separator
            "|",
            # print answer
            output['answer']
        )