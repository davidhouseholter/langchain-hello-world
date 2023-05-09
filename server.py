import os
from dotenv import load_dotenv

import asyncio
from typing import Any, Dict, List, Optional
from langchain import LLMChain, PromptTemplate
from langchain.schema import LLMResult
from grpclib.utils import graceful_exit
from grpclib.server import Server, Stream
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.vectorstores import Chroma

from proto.seq2seq_pb2 import PromptRequest, PromptReply
from proto.seq2seq_grpc import Seq2SeqBase
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain

from langchain.llms import OpenAI
from langchain.callbacks.base import AsyncCallbackHandler
from langchain.callbacks.manager import (
    AsyncCallbackManager,
    AsyncCallbackManagerForChainRun,
    CallbackManager,
    CallbackManagerForChainRun,
    Callbacks,
)
from langchain.chains.chat_vector_db.prompts import (CONDENSE_QUESTION_PROMPT,
                                                     QA_PROMPT)
from langchain.callbacks.tracers import LangChainTracer
from langchain.chains import ChatVectorDBChain

from langchain.chains.llm import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.vectorstores.base import VectorStore
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.redis import Redis

from services.openai_seq2seq import Seq2SeqService

_host : str = os.getenv('HOST') or '127.0.0.1'
_port : int = os.getenv('PORT') or 50050

async def main(*, host = _host, port =  _port) -> None:
    server = Server([Seq2SeqService()])
    with graceful_exit([server]):
        await server.start(host, port)
        print(f'Serving on {host}:{port}')
        await server.wait_closed()

if __name__ == '__main__':
    load_dotenv()
    asyncio.run(main())


