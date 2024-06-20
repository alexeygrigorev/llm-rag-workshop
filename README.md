# LLM RAG Workshop

Chat with your own data - LLM+RAG workshop

The content here is based on [LLM Zoomcamp](https://github.com/DataTalksClub/llm-zoomcamp) - a free course about the engineering aspects of LLMs. The course just started, you can still enroll. 

If you want to run a similar workshop in your company, contact
me at alexey@datatalks.club.


For this workshop, you need:

- Docker
- Python 3 (we use 3.10)
- OpenAI account (optional)
- GitHub account (optional)
- HuggingFace account (optional)


# Plan

* LLM and RAG (theory)
* Preparing the environment (codespaces)
    * Installing pipenv and direnv
    * Running ElasticSearch
* Indexing and retrieving documents with ElasticSearch
* Generating the answers with OpenAI

Extended workshop:

* Creating a web interface with Streamlit
* Running LLMs locally
    * Replacing OpenAI with Ollama
    * Running Ollama and ElasticSearch in Docker-Compose
* Using Open-Source LLMs from HuggingFace Hub


# LLM and RAG

I generated that with ChatGPT:

## Large Language Models (LLMs)
- **Purpose:** Generate and understand text in a human-like manner.
- **Structure:** Built using deep learning techniques, especially Transformer architectures.
- **Size:** Characterized by having a vast number of parameters (billions to trillions), enabling nuanced understanding and generation.
- **Training:** Pre-trained on large datasets of text to learn a broad understanding of language, then fine-tuned for specific tasks.
- **Applications:** Used in chatbots, translation services, content creation, and more.

## Retrieval-Augmented Generation (RAG)
- **Purpose:** Enhance language model responses with information retrieved from external sources.
- **How It Works:** Combines a language model with a retrieval system, typically a document database or search engine.
- **Process:** 
  - Queries an external knowledge source based on input.
  - Integrates retrieved information into the generation process to provide contextually rich and accurate responses.
- **Advantages:** Improves the factual accuracy and relevance of generated text.
- **Use Cases:** Fact-checking, knowledge-intensive tasks like medical diagnosis assistance, and detailed content creation where accuracy is crucial.

Use ChatGPT to show the difference between generating and RAG.


What we will do: 

* Index Zoomcamp FAQ documents
    * DE Zoomcamp: https://docs.google.com/document/d/19bnYs80DwuUimHM65UV3sylsCn2j1vziPOwzBwQrebw/edit
    * ML Zoomcamp: https://docs.google.com/document/d/1LpPanc33QJJ6BSsyxVg-pWNMplal84TdZtq10naIhD8/edit
    * MLOps Zoomcamp: https://docs.google.com/document/d/12TlBfhIiKtyBv8RnsoJR6F72bkPDGEvPOItJIxaEzE0/edit
* Create a Q&A system for answering questions about these documents 


# Preparing the Environment 

We will use codespaces - but it will work in any environment with Docker and Python 3

In codespaces:

* Create a repository, e.g. "llm-zoomcamp-rag-workshop"
* Start a codespace there

We will use pipenv for dependency management. It's optional but strongly recommended if you're doing the workshop locally, and not on codespaces.

Let's install it: 

```bash
pip install pipenv
```

Install the packages:

```bash
pipenv install tqdm notebook==7.1.2 openai elasticsearch
```

If you use OpenAI, we need the key:

* Sign up at https://platform.openai.com/ if you don't have an account
* Go to https://platform.openai.com/api-keys
* Create a new key, copy it 

Let's put the key to an env variable:


```bash
export OPENAI_API_KEY="TOKEN"
```

But a better way for managing keys is using direnv:

```bash
sudo apt update
sudo apt install direnv 
direnv hook bash >> ~/.bashrc
```

Create / edit `.envrc` in your project directory:

```bash
export OPENAI_API_KEY='sk-proj-key'
```

Make sure `.envrc` is in your `.gitignore` - never commit it!

```bash
echo ".envrc" >> .gitignore
```

Allow direnv to run:

```bash
direnv allow
```

Start a new terminal, and there run jupyter:


```bash
pipenv run jupyter notebook
```

In another terminal, run elasticsearch with docker:

```bash
docker run -it \
    --rm \
    --name elasticsearch \
    -p 9200:9200 \
    -p 9300:9300 \
    -e "discovery.type=single-node" \
    -e "xpack.security.enabled=false" \
    docker.elastic.co/elasticsearch/elasticsearch:8.4.3
```

Verify that ES is running

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

# Retrieval

RAG consists of multiple components, and the first is R - "retrieval". For retrieval, we need a search system. In our example, we will use elasticsearch for searching. 

## Searching in the documents

Create a nootebook "elastic-rag" or something like that. We will use it for our experiments

First, we need to download the docs:

```bash
wget https://github.com/alexeygrigorev/llm-rag-workshop/raw/main/notebooks/documents.json
```

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

You should see the same response as earlier with `curl`.

Before we can index the documents, we need to create an index (an index in elasticsearch is like a table in a "usual" databases):

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

Now we're ready to index all the documents:

```python
from tqdm.auto import tqdm

for doc in tqdm(documents):
    es.index(index=index_name, document=doc)
```



## Retrieving the docs

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

This query:

* Retrieves top 5 matching documents.
* Searches in the "question", "text", "section" fields, prioritizing "question" using `multi_match` query with type `best_fields` (see [here](https://github.com/DataTalksClub/llm-zoomcamp/blob/main/01-intro/elastic-search.md) for more information)
* Matches user query "How do I join the course after it has started?".
* Shows results only for the "data-engineering-zoomcamp" course.

Let's see the output:

```python
response = es.search(index=index_name, body=search_query)

for hit in response['hits']['hits']:
    doc = hit['_source']
    print(f"Section: {doc['section']}")
    print(f"Question: {doc['question']}")
    print(f"Answer: {doc['text'][:60]}...\n")
```

## Cleaning it

We can make it cleaner by putting it into a function:

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
    print(f"Section: {doc['section']}")
    print(f"Question: {doc['question']}")
    print(f"Answer: {doc['text'][:60]}...\n")
```

# Generation - Answering questions

Now let's do the "G" part - generation based on the "R" output

## OpenAI

Today we will use OpenAI (it's the easiest to get started with). In the course, we will learn how to use open-source models 

Make sure we have the SDK installed and the key is set.

This is how we communicate with ChatGPT3.5:

```python
from openai import OpenAI

client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "The course already started. Can I still join?"}]
)
print(response.choices[0].message.content)
```

## Building a Prompt

Now let's build a prompt. First, we put all the 
documents together in one string:


```python
context_template = """
Section: {section}
Question: {question}
Answer: {text}
""".strip()

context_docs = retrieve_documents(user_question)

context_result = ""

for doc in context_docs:
    doc_str = context_template.format(**doc)
    context_result += ("\n\n" + doc_str)

context = context_result.strip()
print(context)
```

Now build the actual prompt:

```python
prompt = f"""
You're a course teaching assistant. Answer the user QUESTION based on CONTEXT - the documents retrieved from our FAQ database. 
Only use the facts from the CONTEXT. If the CONTEXT doesn't contan the answer, return "NONE"

QUESTION: {user_question}

CONTEXT:

{context}
""".strip()
```

Now we can put it to OpenAI API:

```python
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": prompt}]
)
answer = response.choices[0].message.content
answer
```

Note: there are system and user prompts, we can also experiment with them to make the design of the prompt cleaner.

## Cleaning

Now let's put everything together in one function:


```python
context_template = """
Section: {section}
Question: {question}
Answer: {text}
""".strip()

prompt_template = """
You're a course teaching assistant.
Answer the user QUESTION based on CONTEXT - the documents retrieved from our FAQ database.
Don't use other information outside of the provided CONTEXT.  

QUESTION: {user_question}

CONTEXT:

{context}
""".strip()


def build_context(documents):
    context_result = ""
    
    for doc in documents:
        doc_str = context_template.format(**doc)
        context_result += ("\n\n" + doc_str)
    
    return context_result.strip()


def build_prompt(user_question, documents):
    context = build_context(documents)
    prompt = prompt_template.format(
        user_question=user_question,
        context=context
    )
    return prompt

def ask_openai(prompt, model="gpt-4o"):
    response = client.chat.completions.create(
        model=model,
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

Now we can ask it different questions

```python
qa_bot("I'm getting invalid reference format: repository name must be lowercase")
```

```python
qa_bot("I can't connect to postgres port 5432, my password doesn't work")
```

```python
qa_bot("how can I run kafka?")
```


# What's next

* Use Open-Souce
* Build an interface, e.g. streamlit
* Deploy it

# Extended version

For an extended version of this workshop, we will

* Build a UI with streamlit
* Experiment with open-source LLMs and replace OpenAI

# Streamlit UI

We can build simple UI apps with streamlit. Let's install it

```bash
pipenv install streamlit
```

If you want to learn more about streamlit, you can
use [this material](https://github.com/DataTalksClub/project-of-the-week/blob/main/2022-08-14-frontend.md).


We need a simple form with

* Input box for the prompt
* Button
* Text field to display the response (in markdown)

```python
import streamlit as st

def qa_bot(prompt):
    import time
    time.sleep(2)
    return f"Response for the prompt: {prompt}"

def main():
    st.title("DTC Q&A System")

    with st.form(key='rag_form'):
        prompt = st.text_input("Enter your prompt")
        response_placeholder = st.empty()
        submit_button = st.form_submit_button(label='Submit')

    if submit_button:
        response_placeholder.markdown("Loading...")
        response = qa_bot(prompt)
        response_placeholder.markdown(response)

if __name__ == "__main__":
    main()
```

Let's run it

```bash
streamlit run app.py
```

Now we can replace the function `qa_bot`. Let's create 
a file `rag.py` with the content from the notebook.

You can see the content of the file [here](rag.py).

Also, we add a special dropdown menu to select the course:

```python
courses = [
    "data-engineering-zoomcamp",
    "machine-learning-zoomcamp",
    "mlops-zoomcamp"
]
zoomcamp_option = st.selectbox("Select a zoomcamp", courses)
```


# Open-Source LLMs

There are many open-source LLMs. We will use two platforms:

* Ollama for running on CPU
* HuggingFace for running on GPU

## Ollama

The easiest way to run an LLM without a GPU is using [Ollama](https://github.com/ollama/ollama)

Note that the 2 core codespaces instance is not enough.
For this part it's better to create a separate instance
with 4 cores. 

You can also run it locally. I have 8 cores on my laptop,
so it's faster than doing it on codespaces.


Installing for Linux:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Installing for other OS - check the [Ollama website](https://www.ollama.com/download). I successfully tested it on Windows too.

Let's run it:

```bash
ollama start
ollama serve phi3
```

Prompt example:

```
Question: I just discovered the couse. can i still enrol

Context:

Course - Can I still join the course after the start date? Yes, even if you don't register, you're still eligible to submit the homeworks. Be aware, however, that there will be deadlines for turning in the final projects. So don't leave everything for the last minute.

Environment - Is Python 3.9 still the recommended version to use in 2024? Yes, for simplicity (of troubleshooting against the recorded videos) and stability. [source] But Python 3.10 and 3.11 should work fine.

How can we contribute to the course? Star the repo! Share it with friends if you find it useful ❣️ Create a PR if you see you can improve the text or the structure of the repository.

Answer:
```

Ollama's API is compatible with OpenAI's python client, so
we can use it by changing only a few lines of code:

```python
from openai import OpenAI

client = OpenAI(
    base_url='http://localhost:11434/v1/',
    api_key='ollama',
)

response = client.chat.completions.create(
    model='phi3',
    messages=[{"role": "user", "content": prompt}]
)
    
response.choices[0].message.content
```

That's it! Now let's put everything in Docker

## Ollama + Elastic in Docker

We already know how to run Elasticsearch in Docker:

```bash
docker run -it \
    --rm \
    --name elasticsearch \
    -p 9200:9200 \
    -p 9300:9300 \
    -e "discovery.type=single-node" \
    -e "xpack.security.enabled=false" \
    docker.elastic.co/elasticsearch/elasticsearch:8.4.3
```

[This is how we run Ollama in Docker](https://hub.docker.com/r/ollama/ollama):

```bash
docker run -it \
    --rm \
    --name ollama \
    -v ollama:/root/.ollama \
    -p 11434:11434 \
    ollama/ollama
```

When we run it, we need to log in to the container to download
the phi3 model:

```bash
docker exec -it ollama bash

ollama pull phi3
```

After pulling the model, we can query it with OpenAI's python package. Because we do volume mapping, the model files will
stay in the container across multiple runs.

Let's now combine them into one docker-compose file. 

Create a [`docker-compose.yaml`](docker-compose.yaml) file with both Ollama and Elasticsearch.

And now run it:

```bash
docker-compose up
```

## HuggingFace Hub

Ollama can run locally on a CPU. But there are many models 
that require a GPU.

For running them, we will use Colab or other notebook platform with a GPU (for example, SaturnCloud). Let's stop our codespace
for now.

In Colab, you need to enable GPU:

* Create a notebook: https://colab.research.google.com/#create=true
* Runtime -> Change runtime type -> T4 GPU
*  `!nvidia-smi` to verify you have a GPU

Now we need to install the dependencies:

```
!pip install -U transformers accelerate bitsandbytes
```

Also, it's tricky to run Elasticsearch on Colab, so we will replace
it with [minsearch](https://github.com/alexeygrigorev/minsearch) - a simple in-memory search library:

```
!wget https://raw.githubusercontent.com/alexeygrigorev/minsearch/main/minsearch.py
```

Let's get the data and create an index:

```python
import requests

docs_url = 'https://github.com/alexeygrigorev/llm-rag-workshop/raw/main/notebooks/documents.json'
docs_response = requests.get(docs_url)
documents_raw = docs_response.json()

documents = []

for course in documents_raw:
    course_name = course['course']

    for doc in course['documents']:
        doc['course'] = course_name
        documents.append(doc)


import minsearch

index = minsearch.Index(
    text_fields=["question", "text", "section"],
    keyword_fields=["course"]
)

index.fit(documents)
```

Searching with minsearch:

```python
query = "I just discovered the course, can I still join?"

filter_dict = {"course": "data-engineering-zoomcamp"}
boost_dict = {"question": 3}

index.search(query, filter_dict, boost_dict, num_results=5)
```

Let's replace our search function: 

```python
def retrieve_documents(query, max_results=5):
    filter_dict = {"course": "data-engineering-zoomcamp"}
    boost_dict = {"question": 3}

    return index.search(query, filter_dict, boost_dict, num_results=5)
```

We will use Google's FLAN T5 model: [`google/flan-t5-xl`](https://huggingface.co/google/flan-t5-xl). 

Downloading and loading it:


```python
from transformers import T5ForConditionalGeneration, T5Tokenizer

model_name = "google/flan-t5-xl"

model = T5ForConditionalGeneration.from_pretrained(model_name, device_map="auto")
tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
tokenizer.model_max_length = 4096
```

Using it: 

```python
input_text = "translate English to German: How old are you?"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")

outputs = model.generate(input_ids)
print(tokenizer.decode(outputs[0]))
```

Let's put it to a function:

```python
def llm(prompt):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
    outputs = model.generate(input_ids, )
    result = tokenizer.decode(outputs[0])
    return result
```

Everything together:

```python
context_template = """
Section: {section}
Question: {question}
Answer: {text}
""".strip()

prompt_template = """
You're a course teaching assistant.
Answer the user QUESTION based on CONTEXT - the documents retrieved from our FAQ database.
Don't use other information outside of the provided CONTEXT.  

QUESTION: {user_question}

CONTEXT:

{context}
""".strip()


def build_context(documents):
    context_result = ""
    
    for doc in documents:
        doc_str = context_template.format(**doc)
        context_result += ("\n\n" + doc_str)
    
    return context_result.strip()


def build_prompt(user_question, documents):
    context = build_context(documents)
    prompt = prompt_template.format(
        user_question=user_question,
        context=context
    )
    return prompt


def qa_bot(user_question):
    context_docs = retrieve_documents(user_question)
    prompt = build_prompt(user_question, context_docs)
    answer = llm(prompt)
    return answer
```

Making the answers longer:

```python
def llm(prompt, generate_params=None):
    if generate_params is None:
        generate_params = {}

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
    outputs = model.generate(
        input_ids,
        max_length=generate_params.get("max_length", 100),
        num_beams=generate_params.get("num_beams", 5),
        do_sample=generate_params.get("do_sample", False),
        temperature=generate_params.get("temperature", 1.0),
        top_k=generate_params.get("top_k", 50),
        top_p=generate_params.get("top_p", 0.95),
    )
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result
```

Explanation of the parameters:

* `max_length`: Set this to a higher value if you want longer responses. For example, `max_length=300`.
* `num_beams`: Increasing this can lead to more thorough exploration of possible sequences. Typical values are between 5 and 10.
* `do_sample`: Set this to `True` to use sampling methods. This can produce more diverse responses.
* `temperature`: Lowering this value makes the model more confident and deterministic, while higher values increase diversity. Typical values range from 0.7 to 1.5.
* `top_k` and `top_p`: These parameters control nucleus sampling. `top_k` limits the sampling pool to the top `k` tokens, while `top_p` uses cumulative probability to cut off the sampling pool. Adjust these based on the desired level of randomness.


Final notebook:

* [notebooks/google_flan_t5.ipynb](notebooks/google_flan_t5.ipynb)
* [On Colab](https://colab.research.google.com/drive/1ldGq6PJw5_vIEWFcxZsNlxX6IJR-rWPk?usp=sharing)

Other models:

* `microsoft/Phi-3-mini-128k-instruct`
* `mistralai/Mistral-7B-v0.1`
* And many more


# Conclusions

That was fun - thanks!