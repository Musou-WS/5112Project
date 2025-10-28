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

def findAfter(text, marker):
    marker_index = text.find(marker)
    if marker_index != -1:
        start = marker_index + len(marker)
        result = text[start:]
    else:
        result = ""
    return result

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

@app.route('/generateRelationshipTree')
def generateRelationshipTree():
    content = request.args.get('content')
    messages = [
        {
            "role": "user",
            "content": content + '. Generate a relationships between characters based on this.'
        },
    ]
    prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    outputs = pipe(prompt, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
    return findAfter(outputs[0]["generated_text"], '<|assistant|>')

@app.route('/initializeStory')
def initializeStory():
    content = request.args.get('content')
    messages = [
        {
            "role": "user",
            "content": content + '. Generate a story based on this.'
        },
    ]
    prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    outputs = pipe(prompt, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
    return findAfter(outputs[0]["generated_text"], '<|assistant|>')

@app.route('/buildEvidenceChain')
def buildEvidenceChain():
    content = request.args.get('content')
    messages = [
        {
            "role": "user",
            "content": content + '. Is there any unclear chain of evidence in this story? If yes, fill and output them.'
        },
    ]
    prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    outputs = pipe(prompt, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
    return findAfter(outputs[0]["generated_text"], '<|assistant|>')

@app.route('/updateEvidenceChain')
def updateEvidenceChain():
    content = request.args.get('content')
    evidenceChain = request.args.get('evidenceChain')
    messages = [
        {
            "role": "user",
            "content": content + '. Evidence Chain is: ' + evidenceChain + 'Is there any other unclear chain of evidence in this story? If yes, fill and output them.'
        },
    ]
    prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    outputs = pipe(prompt, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
    return findAfter(outputs[0]["generated_text"], '<|assistant|>')

@app.route('/buildUnclearParts')
def buildUnclearParts():
    content = request.args.get('content')
    messages = [
        {
            "role": "user",
            "content": content + '. Is there any unclear part in this story? If yes, fill and output them.'
        },
    ]
    prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    outputs = pipe(prompt, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
    return findAfter(outputs[0]["generated_text"], '<|assistant|>')

@app.route('/updateUnclearParts')
def updateUnclearParts():
    content = request.args.get('content')
    unclearParts = request.args.get('unclearParts')
    messages = [
        {
            "role": "user",
            "content": content + '. Unclear Parts are: ' + unclearParts + 'Is there any other unclear part in this story? If yes, fill and output them.'
        },
    ]
    prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    outputs = pipe(prompt, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
    return findAfter(outputs[0]["generated_text"], '<|assistant|>')

@app.route('/rebuildStory')
def rebuildStory():
    content = request.args.get('content')
    evidenceChain = request.args.get('evidenceChain')
    unclearParts = request.args.get('unclearParts')
    messages = [
        {
            "role": "user",
            "content": content + 'Evidence Chain is: ' + evidenceChain + 'Unclear Parts are: ' + unclearParts + 'Rebuild and output the story.'
        },
    ]
    prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    outputs = pipe(prompt, do_sample=False, temperature=0.7, top_k=50, top_p=0.95)
    return findAfter(outputs[0]["generated_text"], '<|assistant|>')

if __name__ == '__main__':
    app.run(port=3000, debug=True)
