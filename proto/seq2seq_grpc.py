# Generated by the Protocol Buffers compiler. DO NOT EDIT!
# source: proto/seq2seq.proto
# plugin: grpclib.plugin.main
import abc
import typing

import grpclib.const
import grpclib.client
if typing.TYPE_CHECKING:
    import grpclib.server

import proto.seq2seq_pb2


class Seq2SeqBase(abc.ABC):

    @abc.abstractmethod
    async def PromptModel(self, stream: 'grpclib.server.Stream[proto.seq2seq_pb2.PromptRequest, proto.seq2seq_pb2.PromptReply]') -> None:
        pass

    def __mapping__(self) -> typing.Dict[str, grpclib.const.Handler]:
        return {
            '/seq2seq.Seq2Seq/PromptModel': grpclib.const.Handler(
                self.PromptModel,
                grpclib.const.Cardinality.UNARY_STREAM,
                proto.seq2seq_pb2.PromptRequest,
                proto.seq2seq_pb2.PromptReply,
            ),
        }


class Seq2SeqStub:

    def __init__(self, channel: grpclib.client.Channel) -> None:
        self.PromptModel = grpclib.client.UnaryStreamMethod(
            channel,
            '/seq2seq.Seq2Seq/PromptModel',
            proto.seq2seq_pb2.PromptRequest,
            proto.seq2seq_pb2.PromptReply,
        )
