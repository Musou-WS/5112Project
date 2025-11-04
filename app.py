# 1. PowerPoint
# 2. Technical report (2 pages (word file))
# -> environment + API + link to the dataset
# -> required for setting up your system
# 3. Source Code
# 4. Movie (implementation)

# https://strandmag.com/the-magazine/short-stories/

import math
import numpy as np
import json
from flask import Flask, request

# database
import chromadb

# AI agent
from groq import Groq
app = Flask(__name__)

app.config.from_pyfile('config.py')

client = Groq(
    api_key=app.config['GROQ_API_KEY'],
)
chroma_client = chromadb.Client()
# client = chromadb.PersistentClient(path="/path/to/save/to")
collection = chroma_client.create_collection(name="5112")
def insertData():
    with open('data.json', 'r') as f:
        data = json.load(f)
    ids = []
    documents = []
    metadatas = []
    #insert data into database
    for index, item in enumerate(data):
        ids.append(str(index))
        documents.append(item['plots'])
        metadatas.append({"relationship": item['roles']})
    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
    )
insertData()

def simulatedAnnealing(ragRes):
    rels = ragRes['metadatas'][0]
    distances = ragRes['distances'][0]
    largestDistance = distances[-1]
    smallestDistance = distances[0]
    # when largest rel, possibility = 1
    T = largestDistance-smallestDistance # initial temperature
    # when smallest rel, possibility = 0.1
    Tend = -T/math.log(0.1) # final temperature
    Tgap = (T-Tend)/4
    result = []
    for index, rel in enumerate(rels):
        p = math.exp(-(distances[index]-smallestDistance)/T)
        T = T - Tgap
        # generate a random number
        r = np.random.uniform(low=0, high=1)
        # if r pass the possibility test, add the relationship
        if r < p:
            result.append(rel)
    return result

@app.route('/generateRelationships')
def generateRelationships():
    plots = request.args.get('plots').split(',')
    results = []
    for plot in plots:
        ragRes = collection.query(
            query_texts=[plot],  # Chroma will embed this for you
            n_results=5  # how many results to return
        )
        res = []
        rels = simulatedAnnealing(ragRes)
        for rel in rels:
            res.append(rel['relationship'])
        results.append(res)
    return results

@app.route('/generateRelationshipTree')
def generateRelationshipTree():
    content = request.args.get('content')
    messages = [
        {
            "role": "user",
            "content": content + '. Generate a relationship table between characters based on this and create it as a html content without styles.'
        },
    ]
    prompt = client.chat.completions.create(messages=messages, model='openai/gpt-oss-120b')
    return prompt.choices[0].message.content

@app.route('/initializeStory')
def initializeStory():
    content = request.args.get('content')
    messages = [
        {
            "role": "user",
            "content": content + '. Generate a story without subtitle based on this and create it as a html content without styles.'
        },
    ]
    prompt = client.chat.completions.create(messages=messages, model='openai/gpt-oss-120b')
    return prompt.choices[0].message.content

@app.route('/buildEvidenceChain', methods=['POST'])
def buildEvidenceChain():
    json_data = request.get_json()
    content = json_data.get('content')
    messages = [
        {
            "role": "user",
            "content": content + '. Is there any unclear chain of evidence in this story? If yes, fill and output them as a list.'
        },
    ]
    prompt = client.chat.completions.create(messages=messages, model='openai/gpt-oss-120b')
    return prompt.choices[0].message.content

@app.route('/updateEvidenceChain', methods=['POST'])
def updateEvidenceChain():
    json_data = request.get_json()
    content = json_data.get('content')
    evidenceChain = json_data.get('evidenceChain')
    messages = [
        {
            "role": "user",
            "content": 'Content is: ' + content
        },
        {
            "role": "user",
            "content": 'Evidence Chain is: ' + evidenceChain
        },
        {
            "role": "user",
            "content": "Is there any other unclear chain of evidence in this story? If yes, fill and output them as a list."
        }
    ]
    prompt = client.chat.completions.create(messages=messages, model='openai/gpt-oss-120b')
    return prompt.choices[0].message.content

@app.route('/buildUnclearParts', methods=['POST'])
def buildUnclearParts():
    json_data = request.get_json()
    content = json_data.get('content')
    messages = [
        {
            "role": "user",
            "content": content + '. Is there any unclear part in this story? If yes, fill and output them as a list.'
        },
    ]
    prompt = client.chat.completions.create(messages=messages, model='openai/gpt-oss-120b')
    return prompt.choices[0].message.content

@app.route('/updateUnclearParts', methods=['POST'])
def updateUnclearParts():
    json_data = request.get_json()
    content = json_data.get('content')
    unclearParts = json_data.get('unclearParts')
    messages = [
        {
            "role": "user",
            "content": 'Content is: ' + content
        },
        {
            "role": "user",
            "content": 'Unclear Parts are: ' + unclearParts
        },
        {
            "role": "user",
            "content": 'Is there any other unclear part in this story? If yes, fill and output them as a list.'
        },
    ]
    prompt = client.chat.completions.create(messages=messages, model='openai/gpt-oss-120b')
    return prompt.choices[0].message.content

@app.route('/rebuildStory', methods=['POST'])
def rebuildStory():
    json_data = request.get_json()
    content = json_data.get('content')
    evidenceChain = json_data.get('evidenceChain')
    unclearParts = json_data.get('unclearParts')
    messages = [
        {
            "role": "user",
            "content": 'Content is: ' + content
        },
        {
            "role": "user",
            "content": 'Evidence Chain is: ' + evidenceChain
        },
        {
            "role": "user",
            "content": 'Unclear Parts are: ' + unclearParts
        },
        {
            "role": "user",
            "content": 'Add all of the evidence chain and unclear parts to the story and rebuild the story without subtitle as a html content without styles.'
        },
    ]
    prompt = client.chat.completions.create(messages=messages, model='openai/gpt-oss-120b')
    return prompt.choices[0].message.content

if __name__ == '__main__':
    app.run(port=3000, debug=True)
