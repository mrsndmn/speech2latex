# Copyright (2024) Tsinghua University, Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pandas as pd
import tqdm
import torch
from transformers import WhisperFeatureExtractor

from config import Config
from models.salmonn import SALMONN
from utils import prepare_one_sample

import argparse
import wave

def get_wav_duration(file_path):
    with wave.open(file_path, 'r') as wav_file:
        frames = wav_file.getnframes()
        rate = wav_file.getframerate()
        duration = frames / float(rate)
    return duration

parser = argparse.ArgumentParser()
parser.add_argument("--cfg-path", type=str, required=True, help='path to configuration file')
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument(
    "--options",
    nargs="+",
    help="override some settings in the used config, the key-value pair "
    "in xxx=yyy format will be merged into config file (deprecate), "
    "change to --cfg-options instead.",
)

parser.add_argument("--test_table", type=str, required=True)
args = parser.parse_args()

def main(path,promt):

    cfg = Config(args)

    model = SALMONN.from_config(cfg.config.model)
    model.to(args.device)
    model.eval()
    print("Model loaded")

    wav_processor = WhisperFeatureExtractor.from_pretrained(cfg.config.model.whisper_path)

    df = pd.read_csv(path)
    df = df[df["language"] == "eng"]

    dict_output = {
        "gt_latex":[],
        "pr_latex":[],
        "pron":[],
        "audio_path":[]
    }

    for _,row in tqdm.tqdm(df.iterrows(), total=df.shape[0]):
        try:
            wav_path = row["audio_path"]
            gt_latex = row["formula_normalized"]
            pron = row.get("pronunciation", None)
        
            samples = prepare_one_sample(wav_path, wav_processor)
            _prompt = [
                cfg.config.model.prompt_template.format("<Speech><SpeechHere></Speech> " + promt.strip())
            ]

            predict_latex = ""
            with torch.cuda.amp.autocast(dtype=torch.float16):
                predict_latex = model.generate(samples, cfg.config.generate, prompts=_prompt)[0].replace("</s>", "")

            dict_output["gt_latex"].append(gt_latex)
            dict_output["pr_latex"].append(predict_latex)
            dict_output["pron"].append(pron)
            dict_output["audio_path"].append("wav_path")


        except Exception as e:
            print(e)

    pd.DataFrame(dict_output).to_csv("output.csv",index=False)

if __name__ == "__main__":
    prompt = "Recognize the speech and convert the content into text. Any mathematical expressions should be transcribed in LaTeX format."

    print("\nStart working")
    main(args.test_table, prompt)
