import os
import io

from llama_index.llms.ollama import Ollama

from haystack import Pipeline, Document
from haystack.utils import Secret
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.components.generators import OpenAIGenerator
from haystack.components.builders.answer_builder import AnswerBuilder
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack.nodes import PDFReader, FARMReader
from haystack.pipelines import ChainedPipeline, ExtractiveQAPipeline


from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Load the PDF file and create a document store
pdf_file = "../data/llamaindex/Markel_Shareholder_Letter_2022.pdf"
pdf_reader = PDFReader(file_base=pdf_file)
document_store = InMemoryDocumentStore()
document_store.write_documents(pdf_reader.extract_pages())

# Initialize the LLM using Ollama and LlamaIndex
gemma = Ollama(model="gemma:7b", request_timeout=30.0)
qa_pipeline = ExtractiveQAPipeline(document_store, FARMReader(model_name="distilbert-base-nli-mean-tokens"))
retrieval_augmentation_pipeline = ChainedPipeline([qa_pipeline.run.bind(document_store=document_store), gemma.generate.bind(model="text-davinci-003")])

# Create a web page interface for user queries
@app.route('/query', methods=['POST'])
def query():
    user_query = request.json['query']
    docs = qa_pipeline.run(user_query)
    answer = retrieval_augmentation_pipeline.run(user_query, context_documents=docs)[0]
    return jsonify({'answer': answer})

# Create a web page for user to chat
@app.route('/chat', methods=['GET'])
def chat():
    return render_template('chat.html')

if __name__ == '__main__':
    app.run(debug=True)
