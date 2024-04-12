import os
import torch
import torchaudio
import numpy as np
from numpy.typing import NDArray
from hubert import HuBERTManager, CustomHubert, CustomTokenizer
from encodec import EncodecModel
from encodec.utils import convert_audio


class VoiceClone:
    def __init__(
            self,
            model: CustomHubert,
            tokenizer: CustomTokenizer,
            audio_codec: EncodecModel,
            device: str = "cpu",
            max_length: float = 12.0
    ):
        self.device = device
        self.model = model
        self.tokenizer = tokenizer
        self.audio_codec = audio_codec
        # maximum length in seconds
        self.max_length = max_length

    def to(self, device: str):
        self.model.to(device)
        self.tokenizer.to(device)
        self.audio_codec.to(device)
        self.device = device

    @torch.no_grad()
    def __call__(self, wave: NDArray, sample_rate: int):
        """ Extracts prompt from audio wave """
        assert wave.ndim == 2, "Wave must be 2D array"
        # preprocess audio
        ## cut audio if it's too long
        length = wave.shape[1] / sample_rate
        if length > self.max_length:
            wave = wave[:, :int(self.max_length * sample_rate)]
        # convert audio to tensor
        wave = convert_audio(
            wave,
            sample_rate,
            self.audio_codec.sample_rate,
            self.audio_codec.channels
        ).to(self.device)
        # extract semantic vectors & tokens
        semantic_vectors = self.model.forward(
            wave,
            input_sample_hz=self.audio_codec.sample_rate
        )
        semantic_tokens = self.tokenizer.get_token(
            semantic_vectors
        )
        # Extract discrete codes from EnCodec
        encoded_frames = self.audio_codec.encode(wave.unsqueeze(0))
        codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1).squeeze()  # [n_q, T]
        codes = codes.cpu().numpy()
        # move semantic tokens to cpu
        semantic_tokens = semantic_tokens.cpu().numpy()
        return {
            "fine_prompt": codes,
            "coarse_prompt": codes[:2, :],
            "semantic_prompt": semantic_tokens
        }

    @staticmethod
    def setup(
            model_filename: str = "hubert.pt",
            tokenizer_filename: str = "tokenizer.pth",
            install_dir: str = os.path.join("data", "models", "hubert")
    ):
        """ Setup Hubert model and tokenizer """
        hubert_manager = HuBERTManager()
        hubert_manager.make_sure_hubert_installed(
            file_name=model_filename,
            install_dir=install_dir
        )
        hubert_manager.make_sure_tokenizer_installed(
            local_file=tokenizer_filename,
            install_dir=install_dir
        )

    @classmethod
    def from_pretrained(
            cls,
            model_filename: str = "hubert.pt",
            tokenizer_filename: str = "tokenizer.pth",
            install_dir: str = os.path.join("data", "models", "hubert"),
            device: str = "cpu"
    ):
        """ Load pretrained model """
        # pre-load checkpoints
        cls.setup(
            model_filename=model_filename,
            tokenizer_filename=tokenizer_filename,
            install_dir=install_dir
        )
        # load model
        model = CustomHubert(
            checkpoint_path=os.path.join(install_dir, model_filename)
        )
        # load tokenizer
        tokenizer = CustomTokenizer.load_from_checkpoint(
            os.path.join(install_dir, tokenizer_filename)
        )
        # load audio codec
        audio_codec = EncodecModel.encodec_model_24khz()
        audio_codec.set_target_bandwidth(6.0)
        audio_codec.eval()
        # create hubert instance
        hubert = cls(model, tokenizer, audio_codec)
        hubert.to(device)
        return hubert


if __name__ == "__main__":
    import time
    # load pretrained model
    hubert = VoiceClone.from_pretrained()
    # load audio file
    wave, sample_rate = torchaudio.load("/home/sergei/Desktop/female_ru_0001.wav")
    # extract prompt
    start = time.time()
    prompt = hubert(wave, sample_rate)
    print("Time elapsed:", time.time() - start)
    # print(prompt)
