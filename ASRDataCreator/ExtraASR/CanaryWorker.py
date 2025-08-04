from nemo.collections.asr.models import EncDecMultiTaskModel
import torch

class CanaryWorker(torch.nn.Module):
    def __init__(self,device ,beam_size: int = 1):
        super(CanaryWorker, self).__init__()
        self.device = device
        self.model = EncDecMultiTaskModel.from_pretrained('nvidia/canary-1b').to(self.device)
        self.model.eval()

        decode_cfg = self.model.cfg.decoding
        decode_cfg.beam.beam_size = beam_size
        self.model.change_decoding_strategy(decode_cfg)
        
    def get_transcription(self, audio_path):

        predicted_text = self.model.transcribe(
            paths2audio_files=[audio_path],
            batch_size=16,
        )
        return predicted_text[0]