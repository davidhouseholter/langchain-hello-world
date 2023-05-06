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

"""Schemas for the chat app."""
from pydantic import BaseModel, validator
# vectorstore: Optional[VectorStore] = None



class ChatResponse(BaseModel):
    """Chat response schema."""

    sender: str
    message: str
    type: str

    @validator("sender")
    def sender_must_be_bot_or_you(cls, v):
        if v not in ["bot", "you"]:
            raise ValueError("sender must be bot or you")
        return v

    @validator("type")
    def validate_message_type(cls, v):
        if v not in ["start", "stream", "end", "error", "info"]:
            raise ValueError("type must be start, stream or end")
        return v



class MyStreamingLLMCallbackHandler(AsyncCallbackHandler):
    """Callback handler for streaming LLM responses."""
   
    def __init__(self, stream: Stream[PromptRequest, PromptReply]):
        self.stream = stream
    
    # async def on_llm_start(
    #     self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    # ) -> None:
    #     """Run when chain starts running."""
    #     print("zzzz....")
    #     await asyncio.sleep(0.3)
    #     class_name = serialized["name"]
    #     print("Hi! I just woke up. Your llm is starting")

    # async def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
    #     print("zzzz....")

    async def on_llm_new_token(self, token: str, **kwargs: any) -> None:
        print(token)
        await self.stream.send_message(
            PromptReply(message=f'{token}'))
 

class Seq2SeqService(Seq2SeqBase):

    # UNARY_STREAM - response streaming RPC
    async def PromptModel(
        self,
        stream: Stream[PromptRequest, PromptReply],
    ) -> None:
        request = await stream.recv_message()
        assert request is not None
        print(request.prompt)
        await stream.send_message(
            PromptReply(message=f'Starting {request.prompt} \n with the template: \n What is a good name for a company that makes ? List 5"' ))
        #embeddings = OpenAIEmbeddings()
        # rds = Redis(embedding_function=embeddings.embed_query,
        #              redis_url="redis://localhost:6379",  index_name='link')
        # print(rds.index_name)
        stream_handler = MyStreamingLLMCallbackHandler(stream)

        manager = AsyncCallbackManager([stream_handler])
        streaming_llm = OpenAI(
            streaming=True,
            callback_manager=manager,
            verbose=True,
            max_tokens=150,
            temperature=0.9
        )
        prompt = PromptTemplate(
            input_variables=["product"],
            template="What is a good name for a company that makes {product}? List 5",
        )
        llm_chain = LLMChain(
            llm=streaming_llm,
            output_key="json_string",
            prompt=prompt
        )

        resp = await llm_chain.arun(product=request.prompt)
        print(resp)

    
async def main(*, host: str = '127.0.0.1', port: int = 50050) -> None:
    server = Server([Seq2SeqService()])
    with graceful_exit([server]):
        await server.start(host, port)
        print(f'Serving on {host}:{port}')
        await server.wait_closed()


if __name__ == '__main__':
    load_dotenv()

    asyncio.run(main())


