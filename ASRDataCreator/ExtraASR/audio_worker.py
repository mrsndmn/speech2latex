import torch
from transformers import WhisperProcessor, AutoModelForSpeechSeq2Seq
from utils_vad import read_audio

class AudioWorker:

    def __init__(self, whisper_path,model_name, device):

        self.device = device if torch.cuda.is_available() else "cpu"
        print(f"Set up on {self.device}")

        # model_name = f"openai/whisper-{model_name}"

        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            whisper_path,
            device_map="auto",
            # trust_remote_code=True
        ).to(self.device)

        self.processor = WhisperProcessor.from_pretrained(model_name)


        assert torch.cuda.is_available(), "No cuda"

    def get_word_timestamps_batch(self, audios, batch_size):
        results = []

        try:
            audio_arrays = [read_audio(path).detach().cpu().numpy() for path in audios]

            input_features = [self.processor(audio, return_tensors="pt", sampling_rate=16000).input_features for audio in audio_arrays]

            input_features = torch.cat(input_features).to(self.device)

            with torch.no_grad():
                generated_ids = self.model.generate(input_features)

            transcriptions = self.processor.batch_decode(generated_ids, skip_special_tokens=True)

            for transcription in transcriptions:
                results.append({"text": transcription})

            return results

        except Exception as exp:
            raise exp
            return []

