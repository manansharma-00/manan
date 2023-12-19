# pip -q install langchain pypdf sentence_transformers chromadb

from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import DirectoryLoader
loader = DirectoryLoader('./Pdf_files/',glob='./*.pdf',loader_cls = PyPDFLoader)
documents = loader.load()

from langchain.document_loaders import PyPDFDirectoryLoader
loader2 = PyPDFDirectoryLoader("./Pdf_files/")
documents2 = loader2.load()
documents2

from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap  = 20,
)
texts = text_splitter.split_documents(documents)
texts2 = text_splitter.split_documents(documents2)
texts[3]

from langchain.embeddings import HuggingFaceBgeEmbeddings

model_name = "BAAI/bge-small-en"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": True}
hf = HuggingFaceBgeEmbeddings(
    model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
)

from langchain.vectorstores import Chroma
persist_directory = "db"

vectordb = Chroma.from_documents(documents = texts,embedding = hf,persist_directory=persist_directory)

vectordb.persist()
vectordb = None

vectordb = Chroma(persist_directory=persist_directory, embedding_function = hf)

retriever = vectordb.as_retriever(search_kwargs={"k":5})

## Default LLaMA-2 prompt style
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

def get_prompt(instruction, new_system_prompt ):
    SYSTEM_PROMPT = B_SYS + new_system_prompt + E_SYS
    prompt_template =  B_INST + SYSTEM_PROMPT + instruction + E_INST
    return prompt_template

sys_prompt = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible using the context text provided. Your answers should only answer the question once and not have any text after the answer is done.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information. """

instruction = """CONTEXT:/n/n {context}/n

Question: {question}"""
get_prompt(instruction, sys_prompt)

from langchain.prompts import PromptTemplate
prompt_template = get_prompt(instruction, sys_prompt)

llama_prompt = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

from langchain.llms import Together

llama2_llm = Together(
    model="togethercomputer/llama-2-70b-chat",
    temperature=0.7,
    max_tokens=1024,
    together_api_key="d8ec7106bd0c268bf4672dba83272b86054fbe849eba82f3f75ceb17e6d57eb0"
)

from langchain.retrievers.multi_query import MultiQueryRetriever
multiretriever = MultiQueryRetriever.from_llm(
    retriever= retriever, llm=llama2_llm
)

unique_docs = multiretriever.get_relevant_documents(query="Under what conditions will aicte reject my approval")
len(unique_docs)

from langchain.chains import RetrievalQA

qa_chain = RetrievalQA.from_chain_type(llm= llama2_llm, chain_type_kwargs = {"prompt": llama_prompt},chain_type="stuff",retriever= retriever,return_source_documents = True)

# def process_llm_response(llm_response):
#   print(llm_response['result'])
#   print('\n\nSources:')
#   for source in llm_response['source_documents']:
#     print("Page number - "+str(source.metadata['page'])+" "+source.metadata['source'])

# process_llm_response(qa_chain("Under what conditions will aicte reject my approval"))



def process_llm_response(llm_response):
  
  ans1 = ""
  ans1 += llm_response['result']
  ans1 += '\n\nSources:'
  for source in llm_response['source_documents']:
    ans1 += "Page number - "+str(source.metadata['page'])+" "+source.metadata['source']
  return ans1

    


def flask_final_func(Query):
   ans = process_llm_response(qa_chain(Query))
   return ans









