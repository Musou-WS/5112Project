# 1. PowerPoint
# 2. Technical report (2 pages (word file))
# -> environment + API + link to the dataset
# -> required for setting up your system
# 3. Source Code
# 4. Movie (implementation)

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
collection.add(
    ids=["1", "2", "3"],
    documents=[
        "A murdered B",
        "detective C",
        "A killed B"
    ],
    metadatas=[{"relationship": "friends"}, {"relationship": "son"}, {"relationship": "son"}]
)

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
            "content": content + '. Generate a relationship table with details between characters based on this and create it as a html content without styles.'
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
            "content": 'Add all of the evidence chain and unclear parts to the story and rebuild the story as a html content without styles.'
        },
    ]
    prompt = client.chat.completions.create(messages=messages, model='openai/gpt-oss-120b')
    return prompt.choices[0].message.content

if __name__ == '__main__':
    app.run(port=3000, debug=True)
