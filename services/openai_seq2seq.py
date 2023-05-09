

from proto.seq2seq_grpc import Seq2SeqBase
from grpclib.server import Stream
from proto.seq2seq_pb2 import PromptRequest, PromptReply

from langchain import LLMChain, PromptTemplate
from grpclib.server import Stream

from proto.seq2seq_pb2 import PromptRequest, PromptReply
from proto.seq2seq_grpc import Seq2SeqBase

from langchain.llms import OpenAI
from langchain.callbacks.base import AsyncCallbackHandler
from langchain.callbacks.manager import (
    AsyncCallbackManager,
)

from langchain.chains.llm import LLMChain
from langchain.llms import OpenAI
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
        # print(token)
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
        # print(request.prompt)
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
