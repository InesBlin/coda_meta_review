import os
from src.settings import API_KEY_GPT
from llama_index.llms.openai import OpenAI
from llama_index.core.output_parsers import LangchainOutputParser
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import SimpleDirectoryReader, get_response_synthesizer
from llama_index.core import DocumentSummaryIndex
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

os.environ["OPENAI_API_KEY"] = API_KEY_GPT

FOLDER = "./data/rag/documents/h_var_mod_es_d"
INPUT_FILES = [os.path.join(FOLDER, x) for x in os.listdir(FOLDER) if x.endswith(".txt")][:5]

docs = SimpleDirectoryReader(
    input_files=INPUT_FILES
).load_data()
for index, path in enumerate(INPUT_FILES):
    doc = SimpleDirectoryReader(
        input_files=[path]
    ).load_data()
    doc[0].doc_id = str(index)
    docs.extend(doc)


response_schemas = [
    ResponseSchema(
        name="iv",
        description="Independent variable",
    ),
    ResponseSchema(
        name="cat_t1",
        description="Independent variable value 1",
    ),
    ResponseSchema(
        name="cat_t2",
        description="Independent variable value 2",
    ),
    ResponseSchema(
        name="mod",
        description="Moderator",
    ),
    ResponseSchema(
        name="mod_t1",
        description="Moderator Value 1",
    ),
    ResponseSchema(
        name="mod_t2",
        description="Moderator Value 2",
    ),
    ResponseSchema(
        name="effect",
        description="Effect",
    ),
]
lc_output_parser = StructuredOutputParser.from_response_schemas(
    response_schemas
)
output_parser = LangchainOutputParser(lc_output_parser)

chatgpt = OpenAI(temperature=0, model="gpt-3.5-turbo-0125", output_parser=output_parser)
splitter = SentenceSplitter(chunk_size=1024)

response_synthesizer = get_response_synthesizer(
    response_mode="tree_summarize", use_async=True
)
doc_summary_index = DocumentSummaryIndex.from_documents(
    docs,
    llm=chatgpt,
    summary_query='Please extract the following components of the scientific claim in the article: independent variable, independent variable value 1, independent variable value 1, moderator, moderator value 1, moderator value 2, effect',  
    transformations=[splitter],
    response_synthesizer=response_synthesizer,
    show_progress=True,
)

print(doc_summary_index.get_document_summary("0"))

query_engine = doc_summary_index.as_query_engine()

query = """
It is possible to extract one hypothesis per document, that is formatted as follows, according to the information you extracted:
```hypothesis
When comparing studies where {iv} is {cat_t1} and studies where {iv} is {cat_t2}, effect sizes from studies involving {mod_t1} as {mod} are significantly {effect} than effect sizes based on {mod_t2} as {mod}."
````

From all the hypotheses that you have, you must extract the 5 most coherent hypotheses, ranked by decreasing coherence.
"""
response = query_engine.query(query)

doc_summary_index.storage_context.persist(persist_dir="./experiments/rag")
from llama_index.core import StorageContext, load_index_from_storage

storage_context = StorageContext.from_defaults(persist_dir="./experiments/rag")
index = load_index_from_storage(storage_context)