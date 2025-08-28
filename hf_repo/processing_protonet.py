from transformers import SequenceFeatureExtractor
from transformers.utils import PaddingStrategy
from transformers.feature_extraction_utils import BatchFeature
from torchaudio import transforms
from typing import Union
import numpy as np
import torch


class AudioProtoNetFeatureExtractor(SequenceFeatureExtractor):
    _auto_class = "AutoFeatureExtractor"
    model_input_names = ["input_values"]

    def __init__(self,
                 # spectrogram
                 n_fft: int = 2048,
                 feature_size: int = 1,
                 hop_length: int = 256,
                 power: float = 2.0,

                 # mel scale
                 n_mels: int = 256,
                 sampling_rate: int = 32_000,
                 n_stft: int = 1025,

                 # power to db
                 stype: str = "power",
                 top_db: int = 80,

                 # normalization
                 mean: float = -13.369,
                 std: float = 13.162,
                 padding_value: float = 0.0,

                 return_attention_mask: bool = True,
                 **kwargs,
                 ):
        super().__init__(feature_size=feature_size, sampling_rate=sampling_rate, padding_value=padding_value, **kwargs)

        # Store parameters for serialization
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.power = power
        self.n_mels = n_mels
        self.sampling_rate = sampling_rate
        self.n_stft = n_stft
        self.stype = stype
        self.top_db = top_db
        self.mean = mean
        self.std = std
        self.padding_value = padding_value
        self.return_attention_mask = return_attention_mask
        self.spec_transform = None
        self.mel_scale = None
        self.db_scale = None

    def _init_transforms(self): # TODO post init method?
        self.spec_transform = transforms.Spectrogram(n_fft=self.n_fft, hop_length=self.hop_length, power=self.power)
        self.mel_scale = transforms.MelScale(n_mels=self.n_mels, sample_rate=self.sampling_rate, n_stft=self.n_stft)
        self.db_scale = transforms.AmplitudeToDB(stype=self.stype, top_db=self.top_db)

    def __call__(self,
                 waveform_batch: Union[np.ndarray, list[float], list[np.ndarray], list[list[float]]],
                 padding: Union[bool, str, PaddingStrategy] = "longest",
                 max_length: int | None = None,
                 truncation: bool = True,
                 return_tensors: str = "pt"
                 ):
        if self.spec_transform is None:
            self._init_transforms()
        clip_duration = 5 # TODO this is the clip duration used in training
        max_length = max_length or int(int(self.sampling_rate) * clip_duration)

        if isinstance(waveform_batch, (list, np.ndarray)) and not isinstance(waveform_batch[0], (list, np.ndarray)):
            waveform_batch = [waveform_batch]

        waveform_batch = BatchFeature({"input_values": waveform_batch})

        waveform_batch = self.pad(
            waveform_batch,
            padding=padding,
            max_length=max_length,
            truncation=truncation,
            return_attention_mask=self.return_attention_mask
        )
        waveform_batch = waveform_batch["input_values"]
        audio_tensor = torch.as_tensor(waveform_batch)
        spec_gram = self.spec_transform(audio_tensor)
        mel_spec = self.mel_scale(spec_gram)
        mel_spec = self.db_scale(mel_spec)
        mel_spec_norm = (mel_spec - self.mean) / self.std

        return mel_spec_norm.unsqueeze(1)



