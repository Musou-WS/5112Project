# 1. PowerPoint
# 2. Technical report (2 pages (word file))
# -> environment + API + link to the dataset
# -> required for setting up your system
# 3. Source Code
# 4. Movie (implementation)

from flask import Flask, request

# database
import chromadb

# gpt
import torch
from transformers import pipeline
app = Flask(__name__)

app.config.from_pyfile('config.py')
# client = chromadb.CloudClient(
#   api_key=app.config['API_KEY'],
#   tenant=app.config['TENANT'],
#   database=app.config['DATABASE']
# )

chroma_client = chromadb.Client()
# client = chromadb.PersistentClient(path="/path/to/save/to")
collection = chroma_client.create_collection(name="5112")
collection.add(
    ids=["1", "2", "3"],
    documents=[
        "A murdered B",
        "detective C",
        "A killed B"
    ],
    metadatas=[{"relationship": "friends"}, {"relationship": "son"}, {"relationship": "son"}]
)

# gpt
pipe = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.bfloat16, device_map="auto")

@app.route('/generateRelationships')
def generateRelationships():  # put application's code here
    plots = request.args.get('plots').split(',')
    results = []
    for plot in plots:
        ragRes = collection.query(
            query_texts=[plot],  # Chroma will embed this for you
            n_results=2  # how many results to return
        )
        res = []
        for rel in ragRes['metadatas'][0]:
            res.append(rel['relationship'])
        results.append(res)
    return results

@app.route('/ask')
def ask():
    messages = [
        {
            "role": "system",
            "content": "You are a friendly chatbot who always responds in the style of a pirate",
        },
        {
            "role": "user",
            "content": "How many helicopters can a human eat in one sitting?"
        },
    ]
    prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    outputs = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
    return outputs[0]["generated_text"]

if __name__ == '__main__':
    app.run(port=3000, debug=True)
