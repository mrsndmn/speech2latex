import torch
import torchaudio
from transformers import Wav2Vec2Processor, WavLMForCTC

class WavLMWorker(torch.nn.Module):
    def __init__(self,device):
        self.device = device
        super(WavLMWorker, self).__init__()
        self.processor = Wav2Vec2Processor.from_pretrained("patrickvonplaten/wavlm-libri-clean-100h-large")
        self.model = WavLMForCTC.from_pretrained("patrickvonplaten/wavlm-libri-clean-100h-large").to(self.device)

    def load_audio(self, file_path):
        waveform, sample_rate = torchaudio.load(file_path)
        if sample_rate != 16000:
            waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)
        return waveform

    def get_transcription(self, audio_path):
        """
        Метод для получения транскрипции аудиофайла.
        
        audio_path: Путь к аудиофайлу.
        """
        
        # Load audio
        waveform = self.load_audio(audio_path)
        
        # Ensure that the waveform is mono (single channel)
        waveform = waveform.mean(dim=0, keepdim=True)

        # Preprocess the audio
        inputs = self.processor(waveform, sampling_rate=16000, return_tensors="pt", padding=True)
        # Forward pass through WavLM model
        device = next(iter(self.model.parameters())).device
        with torch.no_grad():
            logits = self.model(inputs.input_values.squeeze(0).to(device)).logits

        # Decode the predicted IDs to text
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.decode(predicted_ids[0])
        return transcription
    