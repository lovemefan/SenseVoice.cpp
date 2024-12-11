#!/usr/bin/env python3

from __future__ import annotations

import argparse
import contextlib
import os.path
import sys
from abc import ABC, abstractmethod
from enum import IntEnum, auto
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, ContextManager, Iterator, TypeVar

import numpy as np
import torch

if TYPE_CHECKING:
    from torch import Tensor
try:
    import gguf
except ImportError:
    print("please run command : pip install gguf")


###### MODEL DEFINITIONS ######
class MODEL_ARCH(IntEnum):
    SENSEVOICE_SMALL = auto()
    SENSEVOICE_LARGE = auto()


MODEL_ARCH_NAMES: dict[MODEL_ARCH, str] = {
    MODEL_ARCH.SENSEVOICE_SMALL: "SenseVoiceSmall",
    MODEL_ARCH.SENSEVOICE_LARGE: "SenseVoiceLarge",
}

AnyModel = TypeVar("AnyModel", bound="type[Model]")


class Model(ABC):
    _model_classes: dict[str, type[Model]] = {}

    def __init__(
        self, dir_model: Path, ftype: int, fname_out: Path, is_big_endian: bool
    ):
        self.dir_model = dir_model
        self.ftype = ftype
        self.fname_out = fname_out
        self.is_big_endian = is_big_endian
        self.endianess = (
            gguf.GGUFEndian.BIG if is_big_endian else gguf.GGUFEndian.LITTLE
        )

        self.model_checkpoint = "model.pt"
        self.vad_model_checkpoint = 'silero_vad.pt'
        self.hparams = Model.load_hparams(self.dir_model)
        self.gguf_writer = gguf.GGUFWriter(
            fname_out,
            MODEL_ARCH_NAMES[self.model_arch],
            endianess=self.endianess,
            use_temp_file=False,
        )

    @property
    @abstractmethod
    def model_arch(self) -> MODEL_ARCH:
        pass

    def find_hparam(self, keys: str, default=None, config=None) -> Any:
        """
        Args:
            keys: keys from yaml
            default: default value
            config:
        Return:
            Any: the value of key

        Example:
            the config file of yaml is :
                ```yaml
                    model:
                        frontend_conf:
                            fs: 16000
                            window: hamming
                            n_mels: 80
                ```
            you can get the value of fs by:
                find_hparam('model.frontend_conf.fs', 16000)
        """
        map_key = keys.split(".")
        config = config or self.hparams

        if not isinstance(config, dict):
            return config

        if len(map_key) == 1:
            if map_key[0] not in config.keys():
                print(f"key is not available,using default value:{default}")
            return config.get(map_key[0], default)
        else:
            option = ".".join(map_key[1:])
            return self.find_hparam(
                option, default=default, config=config.get(map_key[0])
            )

    def set_vocab(self):
        raise NotImplementedError

    def get_tensors(self) -> Iterator[tuple[str, Tensor]]:

        print(f"gguf: loading model part '{self.model_checkpoint}'")
        print(f"gguf: loading vad model part '{self.vad_model_checkpoint}'")
        ctx: ContextManager[Any]

        ctx = contextlib.nullcontext(
            torch.load(
                str(self.dir_model / self.vad_model_checkpoint),
                map_location="cpu",
                mmap=True,
                weights_only=True,
            )
        )

        with ctx as model_part:
            for name, data in model_part.items():
                yield name, data

        ctx = contextlib.nullcontext(
            torch.load(
                str(self.dir_model / self.model_checkpoint),
                map_location="cpu",
                mmap=True,
                weights_only=True,
            )
        )

        with ctx as model_part:
            for name, data in model_part.items():
                yield name, data

    def set_gguf_parameters(self):
        raise NotImplementedError

    def write_tensors(self):
        raise NotImplementedError

    def write(self):
        self.write_tensors()
        self.gguf_writer.write_header_to_file()
        self.gguf_writer.write_kv_data_to_file()
        self.gguf_writer.write_tensors_to_file()
        self.gguf_writer.close()

    def write_vocab(self):
        self.gguf_writer.write_header_to_file()
        self.gguf_writer.write_kv_data_to_file()
        self.gguf_writer.close()

    @staticmethod
    def load_hparams(dir_model):
        try:
            import yaml
        except ImportError:
            print("please run: pip install pyyaml")

        with open(dir_model / "config.yaml", "r", encoding="utf-8") as f:
            return yaml.load(f, Loader=yaml.FullLoader)

    @classmethod
    def register(cls, *names: str) -> Callable[[AnyModel], AnyModel]:
        assert names

        def func(modelcls: type[Model]):
            for name in names:
                cls._model_classes[name] = modelcls
            return modelcls

        return func

    @classmethod
    def from_model_architecture(cls, arch):
        try:
            return cls._model_classes[arch]
        except KeyError:
            raise NotImplementedError(f"Architecture {arch!r} not supported!") from None


@Model.register("SenseVoiceSmall")
class SenseVoiceSmall(Model):
    model_arch = MODEL_ARCH.SENSEVOICE_SMALL

    def __init__(self, model_dir, ftype: int, fname_out: Path, is_big_endian: bool):
        super().__init__(model_dir, ftype, fname_out, is_big_endian)
        self.hparams = self.load_hparams(model_dir)
        self.model_name = MODEL_ARCH_NAMES[self.model_arch]

        try:
            import sentencepiece as spm
        except ImportError:
            print("please run `pip install sentencepiece`")

        self.sp = spm.SentencePieceProcessor()
        self.sp.load(os.path.join(model_dir, "chn_jpn_yue_eng_ko_spectok.bpe.model"))

    def set_vocab(self):
        tokens = [self.sp.id_to_piece(i).replace("▁", " ") for i in range(self.sp.vocab_size())]

        self.gguf_writer.add_int32(f"tokenizer.vocab_size", self.sp.vocab_size())

        self.gguf_writer.add_token_list(tokens)

        self.gguf_writer.add_string("tokenizer.unk_symbol", "<unk>")

    def set_gguf_parameters(self):
        # frontend config:
        self.gguf_writer.add_int32(
            "frontend.sample_rate", self.find_hparam("frontend_conf.fs")
        )
        self.gguf_writer.add_string(
            "frontend.window", self.find_hparam("frontend_conf.window")
        )
        self.gguf_writer.add_int32(
            "frontend.num_mels", self.find_hparam("frontend_conf.n_mels")
        )
        self.gguf_writer.add_int32(
            "frontend.frame_length", self.find_hparam("frontend_conf.frame_length")
        )
        self.gguf_writer.add_int32(
            "frontend.frame_shift", self.find_hparam("frontend_conf.frame_shift")
        )
        self.gguf_writer.add_int32(
            "frontend.lfr_m", self.find_hparam("frontend_conf.lfr_m")
        )
        self.gguf_writer.add_int32(
            "frontend.lfr_n", self.find_hparam("frontend_conf.lfr_n")
        )

        # encoder
        self.gguf_writer.add_int32(
            "encoder.output_size", self.find_hparam("encoder_conf.output_size")
        )

        self.gguf_writer.add_int32(
            "encoder.attention_heads", self.find_hparam("encoder_conf.attention_heads")
        )

        self.gguf_writer.add_int32(
            "encoder.linear_units", self.find_hparam("encoder_conf.linear_units")
        )

        self.gguf_writer.add_int32(
            "encoder.num_blocks", self.find_hparam("encoder_conf.num_blocks")
        )

        self.gguf_writer.add_int32(
            "encoder.tp_blocks", self.find_hparam("encoder_conf.tp_blocks")
        )

        self.gguf_writer.add_int32(
            "encoder.kernel_size", self.find_hparam("encoder_conf.kernel_size")
        )

        self.gguf_writer.add_int32(
            f"encoder.sanm_shfit", self.find_hparam("encoder_conf.sanm_shfit")
        )

    def write_one_tensor(self, data_torch, name):
        old_dtype = data_torch.dtype

        # convert any unsupported data types to float32
        if data_torch.dtype not in (torch.float16, torch.float32):
            data_torch = data_torch.to(torch.float32)

        _data = data_torch.numpy()
        # use max to avoid n_dim of single tensor become 0
        if len(_data.shape) != 0:
            data = _data
        else:
            data = data_torch.numpy()

        n_dims = len(data.shape)
        data_dtype = data.dtype

        # if f32 desired, convert any float16 to float32
        if self.ftype == 0 and data_dtype == np.float16 and 'fsmn_block.weight' not in name:
            data = data.astype(np.float32)

        # TODO: Why cant we use these float16 as-is? There should be not reason to store float16 as float32
        if self.ftype == 1 and data_dtype == np.float16 and n_dims == 1:
            data = data.astype(np.float32)

        # if f16 desired, convert any float32 2-dim weight tensors to float16
        if (
                self.ftype == 1
                and data_dtype == np.float32
                and name.endswith(".weight")
                and n_dims == 2
        ):
            data = data.astype(np.float16)

        if name in [
            '_model.stft.forward_basis_buffer.weight',
            '_model.encoder.0.reparam_conv.weight',
            '_model.encoder.1.reparam_conv.weight',
            '_model.encoder.2.reparam_conv.weight',
            '_model.encoder.3.reparam_conv.weight',
            '_model.decoder.decoder.2.weight'
        ]:
            data = data.astype(np.float16)

        print(
            f"|{name}| n_dims = {n_dims}| {old_dtype} | {data.dtype} | {data.size}|"
        )

        self.gguf_writer.add_tensor(name, data)
        return data.size

    def write_tensors(self):
        tensor_size = 0

        print(
            "| Layer name | n_dims | torch type | gguf type | parameters size|"
        )
        print("|:|:|:|:|:|")
        for name, data_torch in self.get_tensors():
            # we don't need these
            if name.endswith(("Loss", "loss")):
                continue

            if 'linear_q_k_v' in name:

                q_k_v = data_torch.split(data_torch.size(0) // 3)
                tensor_size += self.write_one_tensor(q_k_v[0], name.replace('linear_q_k_v', 'linear_q'))
                tensor_size += self.write_one_tensor(q_k_v[1], name.replace('linear_q_k_v', 'linear_k'))
                tensor_size += self.write_one_tensor(q_k_v[2], name.replace('linear_q_k_v', 'linear_v'))

            elif 'fsmn_block.weight' in name:
                tensor_size += self.write_one_tensor(data_torch.to(torch.float16), name)
            else:
                tensor_size += self.write_one_tensor(data_torch, name)

        print(f"\ntotal size is {tensor_size}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert model to a GGUF compatible file"
    )
    parser.add_argument(
        "--vocab-only",
        action="store_true",
        help="extract only the vocab",
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="path to write to; default: based on input",
    )
    parser.add_argument(
        "--out_type",
        type=str,
        choices=["f32", "f16"],
        default="f32",
        help="output format - use f32 for float32, f16 for float16",
    )
    parser.add_argument(
        "--bigendian",
        action="store_true",
        help="model is executed on big endian machine",
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="directory containing model file",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dir_model = args.model

    if not dir_model.is_dir():
        print(f"Error: {args.model} is not a directory", file=sys.stderr)
        sys.exit(1)

    ftype_map = {
        "f32": gguf.GGMLQuantizationType.F32,
        "f16": gguf.GGMLQuantizationType.F16,
    }

    if args.output is not None:
        fname_out = args.output
    else:
        # output in the same directory as the model by default
        fname_out = dir_model / f"ggml-model-{args.outtype}.gguf"

    print(f"Loading model: {dir_model.name}")

    hparams = Model.load_hparams(dir_model)

    with torch.inference_mode():
        model_class = Model.from_model_architecture(hparams["model"])
        model_instance = model_class(
            dir_model, ftype_map[args.out_type], fname_out, args.bigendian
        )

        print("Set model parameters")
        model_instance.set_gguf_parameters()

        print("Set model tokenizer")
        model_instance.set_vocab()

        if args.vocab_only:
            print(f"Exporting model vocab to '{fname_out}'")
            model_instance.write_vocab()
        else:
            print(f"Exporting model to '{fname_out}'")
            model_instance.write()

        print(f"Model successfully exported to '{fname_out}'")


if __name__ == "__main__":
    main()
