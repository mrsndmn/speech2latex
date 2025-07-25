import torch
from transformers import  pipeline,WhisperProcessor,WhisperForConditionalGeneration,WhisperTokenizer
from file_worker import write_json,read_json,create_or_pass_dir
from utils_vad import get_speech_timestamps, read_audio
import re
import copyreg
import os

class AudioWorker:

  def __init__(self,model_name,c,pickle_model_whisper = None,pickle_model_vad = None):
    
    self.device = f"cuda:{c}" if torch.cuda.is_available() else "cpu"
    model_name = f"openai/whisper-{model_name}"
    self.pipe = pipeline(
      "automatic-speech-recognition",
      model = model_name,
      # chunk_length_s=30,
      device=self.device,
      torch_dtype=torch.bfloat16
    )

    # self.vad_model, _ = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad')

    # self.pipe.model.share_memory()
    # self.vad_model.share_memory() 

    # copyreg.pickle(type( self.pipe.model), pickle_model_whisper)
    # copyreg.pickle(type(self.vad_model), pickle_model_vad)
    assert  torch.cuda.is_available(), "no cuda(("

        


  def get_words_from_sentence(self,sentence,timestamp_words,start_index):
    sentence =  sentence.split()
    value = ""
    index = 0
    c = 0
    for i,word in enumerate(timestamp_words[start_index:]):
      if(word["text"].strip() == "_"): #
        start_index+=1
        continue
      value+=word["text"].strip()
      if(value == sentence[index]):
        index+=1
        value = ""

      c+=1
              
      if index == len(sentence):
        break

    return timestamp_words[start_index:start_index + c], start_index + c
  

  def change_previos_value(self,p_value,word_timestamp,_res,len_res):
    p_value["text"] = p_value["text"] + word_timestamp["text"]
    p_value["timestamp"] = (p_value["timestamp"][0] ,word_timestamp["timestamp"][1])
    _res[len_res - 1] = p_value


  def get_word_timestamps(self,audio_path):
    wav = read_audio(audio_path)
    audio_arr = wav.detach().cpu().numpy()

    try:
      res = self.pipe(audio_arr, batch_size=8, return_timestamps="word")["chunks"]
      return res
    except Exception as exp:
      print(exp)
      return []
      

  def get_word_timestamps_batch(self,audios,batch_size):
    results = []
    try:
      audios = [read_audio(os.path.join(*path)).detach().cpu().numpy() for path in audios]
      result = self.pipe(audios,batch_size = batch_size ,return_timestamps="word")
      results.extend(result)
      return results
    except Exception as exp:
      print(exp)
      return []


  def get_word_timestamps_by_vad(self,audio_path):
    num_digits = 8

    
    wav = read_audio(audio_path)
    timestamps = get_speech_timestamps(wav,self.vad_model)

    result = []
    end_audio = len(wav)

    if(len(timestamps)==0):
        result = [{"text":"_","timestamp":(0,round(end_audio/16000,num_digits))}]
        return result


    lt = len(timestamps)
    for i in range(lt):
        k = 2.5

        _s = timestamps[i]["start"]
        _e = timestamps[i]["end"]

        s = round(_s/16000,num_digits)
        e = round(_e/16000,num_digits) 

        _next_s = timestamps[(i+1)%lt]["start"]
        next_s = round(_next_s/16000,num_digits)
		
        if k * 16000 > (_next_s - _e) > 0:
          e = next_s
          _e = _next_s
        
        _wav = wav[_s:_e]
        audio_arr = _wav.detach().cpu().numpy()

        res = None
        try:
            res = self.pipe(audio_arr, batch_size=8, return_timestamps="word")["chunks"]
        except Exception as exp:
            result.append({"text":"_","timestamp":(e,next_s)})
            continue
        
        for i in range(len(res)):
          word_timestamp = {}

          _s = round(res[i]["timestamp"][0],num_digits)
          _e = res[i]["timestamp"][1]
          
          _s+=s
          _e = round(end_audio/16000,num_digits) if _e == None else round( _e+s, num_digits) 

          word_timestamp["text"] = res[i]["text"]
          word_timestamp["timestamp"] = (_s,_e)

          res[i] = word_timestamp

        result.extend(res)

        if(next_s - e) > k:
          result.append({"text":"_","timestamp":(e,next_s)})
    
    
    if(timestamps[-1]["end"] < end_audio ):
        result.append({"text":"_","timestamp":(
          round(timestamps[-1]["end"]/16000,num_digits),
					round(end_audio/16000,num_digits)
				)})
    
    return result

        
  def get_sentences_with_timestamps(self, timestamps_words):
    if len(timestamps_words) == 0:
      print("Length of timestamps_words is 0")
      return None
    
    if(len(timestamps_words) != 0 and timestamps_words[-1]["timestamp"][1] == None):
      timestamps_words = timestamps_words[:-1]

    if len(timestamps_words) == 0:
      print("Length of timestamps_words is 0")
      return None
    
    length = len(timestamps_words)
    timestamps_sentences = []
    buffer = []

    MIN_TIME = 2
  
    start_timestamp = timestamps_words[0]["timestamp"][0]
    for i in range(length):
      isNextVoid = False

      tword = timestamps_words[i]
      timestamp = tword["timestamp"]
      word = tword["text"]
      
      if (i<length - 1 and timestamps_words[i + 1]["text"] == "_" and not word.endswith('.')):
        isNextVoid = True

      buffer.append(word)
      if((word.endswith('.') or word.endswith('?') or word.endswith('!') or word.endswith('_') or isNextVoid) and (timestamp[1] - start_timestamp) >= MIN_TIME ):
        timestamps_sentences.append({'text': "".join(buffer), 'timestamp': (start_timestamp,timestamp[1])})
        start_timestamp = timestamps_words[(i+1)%length]["timestamp"][0]
        buffer = []

    if(len(buffer)!=0):
      # if (timestamp[1] - start_timestamp) >= MIN_TIME:
      timestamps_sentences.append({'text': "".join(buffer), 'timestamp': (start_timestamp,timestamp[1])})
      # else:
        # if len(timestamps_sentences) == 0:
        #   print("Length of timestamps_sentences is 0 after preparing ")
        #   return None

        # last_text = timestamps_sentences[-1]["text"]
        # last_timestamp = timestamps_sentences[-1]["timestamp"]
        # timestamps_sentences[-1] = {'text': last_text + "".join(buffer), 'timestamp': (last_timestamp[0],timestamp[1])}


    if len(timestamps_sentences) == 0:
      print("Length of timestamps_sentences is 0")
      return None
    
    return timestamps_sentences

  def get_transcription_lang(self,waveform):
    input_features = self.processor(waveform, return_tensors="pt", sampling_rate=16000).input_features
    lang_token = self.model.generate(input_features, max_new_tokens=1)[0,1]
    language_code = self.tokenizer.decode(lang_token) 

    return language_code

  def extract_second_token(self,text):

      pattern = re.compile(r'<\|(.{2})\|>')
      match = pattern.search(text)
  
      if match:
          return match.group(1)
      else:
          return None



