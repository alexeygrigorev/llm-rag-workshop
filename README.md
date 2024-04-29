# llm-rag-workshop
Chat with your own data - LLM+RAG workshop

You need: 

- Docker
- Python 3.10
- OpenAI account and/or HuggingFace account



# preparing the env 

using codespaces - but will work anywhere 

* crete a repo
* copy the rest from another workshop


Install pipenv 

```bash
pip install pipenv
```

Install the packages

```bash
pipenv install (content from pipfile)
```

If you use OpenAI, let's put the key to an env variable:

```bash
export OPENAI_API_KEY="TOKEN"
```

Run jupyter (in the same terminal where you run the previous command so it can access the token)

```bash
pipenv run jupyter notebook
```

Run elasticsearch with docker

```bash
docker run -it \
    --name elasticsearch \
    -p 9200:9200 \
    -p 9300:9300 \
    -e "discovery.type=single-node" \
    -e "xpack.security.enabled=false" \
    docker.elastic.co/elasticsearch/elasticsearch:8.4.3

```

Verify that ES is runnign

```bash
curl http://localhost:9200
```

You should get something like this:

```json
{
  "name" : "63d0133fc451",
  "cluster_name" : "docker-cluster",
  "cluster_uuid" : "AKW1gxdRTuSH8eLuxbqH6A",
  "version" : {
    "number" : "8.4.3",
    "build_flavor" : "default",
    "build_type" : "docker",
    "build_hash" : "42f05b9372a9a4a470db3b52817899b99a76ee73",
    "build_date" : "2022-10-04T07:17:24.662462378Z",
    "build_snapshot" : false,
    "lucene_version" : "9.3.0",
    "minimum_wire_compatibility_version" : "7.17.0",
    "minimum_index_compatibility_version" : "7.0.0"
  },
  "tagline" : "You Know, for Search"
}
```

Install ES client

```bash
pipenv install elasticsearch
```


## Searching in the documents

Create a nootebook




Let's load the documents

```python
import json

with open('./documents.json', 'rt') as f_in:
    documents_file = json.load(f_in)

documents = []

for course in documents_file:
    course_name = course['course']

    for doc in course['documents']:
        doc['course'] = course_name
        documents.append(doc)
```

Now we'll index these documents with elastic search

First initiate the connection and check that it's working:

```python
from elasticsearch import Elasticsearch

es = Elasticsearch("http://localhost:9200")
es.info()
```

You should see the same response as earlier with curl

Create index (like table in "usual" databases):

```python
index_settings = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0
    },
    "mappings": {
        "properties": {
            "text": {"type": "text"},
            "section": {"type": "text"},
            "question": {"type": "text"},
            "course": {"type": "keyword"} 
        }
    }
}

index_name = "course-questions"
response = es.indices.create(index=index_name, body=index_settings)

response
```

And index the documents:

```python
from tqdm.auto import tqdm

for doc in tqdm(documents):
    es.index(index=index_name, document=doc)
```



# Retrieving the docs

```python
user_question = "How do I join the course after it has started?"

search_query = {
    "size": 5,
    "query": {
        "bool": {
            "must": {
                "multi_match": {
                    "query": user_question,
                    "fields": ["question^3", "text", "section"],
                    "type": "best_fields"
                }
            },
            "filter": {
                "term": {
                    "course": "data-engineering-zoomcamp"
                }
            }
        }
    }
}
```

TODO: explain the query


Show the output:


```python
response = es.search(index=index_name, body=search_query)

for hit in response['hits']['hits']:
    doc = hit['_source']
    print(f"Section: {doc['section']}\nQuestion: {doc['question']}\nAnswer: {doc['text']}\n\n")
```

Clean a little

```python
def retrieve_documents(query, index_name="course-questions", max_results=5):
    es = Elasticsearch("http://localhost:9200")
    
    search_query = {
        "size": max_results,
        "query": {
            "bool": {
                "must": {
                    "multi_match": {
                        "query": query,
                        "fields": ["question^3", "text", "section"],
                        "type": "best_fields"
                    }
                },
                "filter": {
                    "term": {
                        "course": "data-engineering-zoomcamp"
                    }
                }
            }
        }
    }
    
    response = es.search(index=index_name, body=search_query)
    documents = [hit['_source'] for hit in response['hits']['hits']]
    return documents
```

And print the answers:

```python
user_question = "How do I join the course after it has started?"

response = retrieve_documents(user_question)

for doc in response:
    print(f"Section: {doc['section']}\nQuestion: {doc['question']}\nAnswer: {doc['text']}\n\n")
```

# Answering questions



## With OpenAI


Make sure we have the SDK installed:

```bash
pipenv install openai
```

Also we need to set the `OPENAI_API_KEY` variable before launching Jupyter

```bash
export OPENAI_API_KEY="YOUR_KEY"
```

This is how we communicate with ChatGPT3.5:

```python
from openai import OpenAI

client = OpenAI()

response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "What's the formula for Energy?"}]
)
print(response.choices[0].message.content)
```


Now let's build a prompt. First, we put all the 
documents together in one string:


```python
context_docs = retrieve_documents(user_question)

context = ""

for doc in context_docs:
    doc_str = f"Section: {doc['section']}\nQuestion: {doc['question']}\nAnswer: {doc['text']}\n\n"
    context += doc_str

context = context.strip()
print(context)
```

Now build an actual prompt:

prompt = f"""
You're a course teaching assistant. Answer the user QUESTION based on CONTEXT - the documents retrieved from our FAQ database. 
Only use the facts from the CONTEXT. If the CONTEXT doesn't contan the answer, return "NONE"

QUESTION: {user_question}

CONTEXT:

{context}
""".strip()

Let's query OpenAI:

```python
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": prompt}]
)
answer = response.choices[0].message.content
answer
```

TODO: add system and user prompts

Now let's put everything together in one function:


```python
def build_context(documents):
    context = ""

    for doc in documents:
        doc_str = f"Section: {doc['section']}\nQuestion: {doc['question']}\nAnswer: {doc['text']}\n\n"
        context += doc_str
    
    context = context.strip()
    return context


def build_prompt(user_question, documents):
    context = build_context(documents)
    return f"""
You're a course teaching assistant.
Answer the user QUESTION based on CONTEXT - the documents retrieved from our FAQ database.
Don't use other information outside of the provided CONTEXT.  

QUESTION: {user_question}

CONTEXT:

{context}
""".strip()

def ask_openai(prompt, model="gpt-3.5-turbo"):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    answer = response.choices[0].message.content
    return answer

def qa_bot(user_question):
    context_docs = retrieve_documents(user_question)
    prompt = build_prompt(user_question, context_docs)
    answer = ask_openai(prompt)
    return answer
```

Now we can ask it:

```python
qa_bot("I'm getting invalid reference format: repository name must be lowercase")

qa_bot("I can't connect to postgres port 5432, my password doesn't work")

qa_bot("how can I run kafka?")
```

## With Open-Source 

If we can't use OpenAI, we can replace it with a self-hosted 
open-source LLM. 

The best place to find these models is the HuggingFace hub. 

Examples of models:

* `mistralai/Mistral-7B-v0.1`
* `HuggingFaceH4/zephyr-7b-alpha`
* `microsoft/DialoGPT-medium`
* `microsoft/Phi-3-mini-128k-instruct`
* and [many more](https://huggingface.co/models?other=LLM) - also see the [leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)

A good getting started tutorial (if you have a GPU): https://huggingface.co/docs/transformers/en/llm_tutorial





If you have a GPU

If you don't have a GPU

...


# Interface 

Streamlit
