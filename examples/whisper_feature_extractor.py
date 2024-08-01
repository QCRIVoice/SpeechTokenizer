
import torch
from datasets import Dataset, Audio
from transformers import AutoFeatureExtractor, WhisperModel
import logging
import soundfile as sf
logger = logging.getLogger("dump_feature")
class WhisperFeatureReader(object):
    def __init__(self, root, ckpt, layer, device):
        self.device = device
        logger.info(f"device = {self.device}")
        self.model = WhisperModel.from_pretrained(ckpt,cache_dir=root).to(self.device).eval()
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(ckpt)
        self.layer = layer  # one-based

    def get_feats(self, audio):
        inputs = self.feature_extractor(audio, sampling_rate=16_000, return_tensors="pt")
        input_features = inputs.input_features.to(self.device)

        self.model.config.output_hidden_states=True  # to obtain the individual layer embeddings
        decoder_input_ids = torch.tensor([[1, 1]]) * self.model.config.decoder_start_token_id

        whisper_embeddings = self.model(input_features, decoder_input_ids=decoder_input_ids.to(self.device))
        return whisper_embeddings.encoder_hidden_states[self.layer -1]
