import os

import pandas as pd
import json
import zstandard
import re
from tqdm.auto import tqdm
import requests

from multiprocessing.pool import Pool


import subprocess

def latex_validate_formula(formula):
    tex_body = "\\documentclass{article}\n% Pass-through \\mathdefault, which is used in non-usetex mode\n% to use the default text font but was historically suppressed\n% in usetex mode.\n\\newcommand{\\mathdefault}[1]{#1}\n\\usepackage{type1cm}\n\\usepackage{type1ec}\n\\usepackage[utf8]{inputenc}\n\\usepackage{amsfonts}\n\\DeclareUnicodeCharacter{2212}{\\ensuremath{-}}\n% geometry is loaded before the custom preamble as \n% convert_psfrags relies on a custom preamble to change the \n% geometry.\n\\usepackage[papersize=72in, margin=1in]{geometry}\n\n% Use `underscore` package to take care of underscores in text.\n% The [strings] option allows to use underscores in file names.\n\\makeatletter\\@ifpackageloaded{underscore}{}{\\usepackage[strings]{underscore}}\\makeatother\n% Custom packages (e.g. newtxtext) may already have loaded \n% textcomp with different options.\n\\makeatletter\\@ifpackageloaded{textcomp}{}{\\usepackage{textcomp}}\\makeatother\n\\pagestyle{empty}\n\\begin{document}\n% The empty hbox ensures that a page is printed even for empty\n% inputs, except when using psfrag which gets confused by it.\n% matplotlibbaselinemarker is used by dviread to detect the\n% last line's baseline.\n\\fontsize{10.0}{12.5}%\n\\ifdefined\\psfrag\\else\\hbox{}\\fi%\n{\\sffamily $ " + formula + " $}%\n\\end{document}"

    tex_file_name = "test.latex"
    with open(tex_file_name, 'w') as f:
        f.write(tex_body)

    bashCommand = "latex -interaction=nonstopmode --halt-on-error " + tex_file_name
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    # print("output", output)
    # print("error", error)
    # print("process.returncode", process.returncode)

    return process.returncode == 0


def decomptress_zst(zst_file_name):
    assert zst_file_name.endswith(".zst")
    decompresssed_file_name = zst_file_name.removesuffix(".zst")

    with open(zst_file_name, "rb") as compressed_file:
        decompressor = zstandard.ZstdDecompressor()

        with open(decompresssed_file_name, "wb") as output_file:
            decompressor.copy_stream(compressed_file, output_file)

    return decompresssed_file_name


def convert_latex_to_plain(text):
    text = re.sub(r'\\hyperref\[.*?\]\{(.*?)\}', r'\1', text)

    text = re.sub(r'\\textit\{(.*?)\}', r'\1', text)

    text = re.sub(r'\\textbf\{(.*?)\}', r'\1', text)

    text = re.sub(r'\\caption\{(.*?)\}', r'\1', text)

    text = re.sub(r'\\cite\{.*?\}', '', text)

    text = re.sub(r'\\cite\[(.*?)\]\{.*?\}', r'\1', text)

    text = re.sub(r'\\label\{.*?\}', '', text)

    text = re.sub(r'\\ref\{.*?\}', '', text)

    text = re.sub(r'\\noindent', '', text)

    text = re.sub(r'\\emph\{(.*?)\}', r'\1', text)

    text = re.sub(r'\\eqref\{.*?\}', 'equation', text)

    text = text.replace('\item', '')
    
    text = text.replace('.~()', '')
    
    text = text.strip()

    return text

def find_inline_math_matches(text):
    inline_math_pattern = r'\$((?:\\\$|[^$])+?)\$'

    matches = []
    for m in re.finditer(inline_math_pattern, text):
        # start_i, end_i, full_match (with $$), formula body (without $$)
        matches.append((m.start(0), m.end(0), m.group(0), m.group(1)))

    return matches

def check_unknown_commands(in_context_math_data):
    text_no_inline_math = in_context_math_data['text']
    for formula_info in reversed(in_context_math_data['formula_info']):
        match_start = formula_info[0]
        match_end = formula_info[1]
        text_no_inline_math = text_no_inline_math[:match_start] + text_no_inline_math[match_end:]

    unknown_command_matches = re.findall(r'\\([a-zA-Z]+)', text_no_inline_math)

    if '$' in text_no_inline_math:
        # print(f"FOUND unmatched inline math: {text_no_inline_math}\n orig: {in_context_math_data['text']}")
        return True

    if len(unknown_command_matches) > 0:
        # print(f"FOUND UNKNOWN COMMAND {unknown_command_matches}: {text_no_inline_math}\n orig: {in_context_math_data['text']}")
        return True

    return False

def check_contains_n_words(text, N=5):
    found_words = 0
    for word in re.split(r'[,\. ]', text):
        if re.match(r'^[A-Za-z]+$', word) is not None:
            found_words += 1
            if found_words >= N:
                return True

    return False


def make_strates(count_formula_symbols):
    if count_formula_symbols < 10:
        return '3-10'
    elif count_formula_symbols < 20:
        return '10-20'
    elif count_formula_symbols < 30:
        return '20-30'
    elif count_formula_symbols < 50:
        return '30-50'

    return '50+'


def process_arxiv_data(arxiv_sentences, max_sentence_character_length=300, min_sentence_character_length=30):

    in_context_math = []

    for item in tqdm(arxiv_sentences, desc="Processing sentences"):
        sentence = item['sentence']
        document_meta = item['meta']
        if len(sentence) > max_sentence_character_length:
            continue

        if '$' not in sentence:
            continue
        
        if len(sentence) < min_sentence_character_length:
            continue

        if not sentence[0].isupper():
            continue

        if sentence[-1] != '.':
            continue

        if '\t' in sentence:
            continue

        if '~,' in sentence:
            continue

        if '``' in sentence:
            continue

        if '()' in sentence:
            continue

        if "$''$" in sentence:
            continue


        sentence = convert_latex_to_plain(sentence)

        if not check_contains_n_words(sentence):
            continue

        if not latex_validate_formula(sentence):
            continue

        # filter unknown commands outside of inline math
        # \\theconjecture
        formula_info = find_inline_math_matches(sentence)
        
        in_context_math_data = {
            "text": sentence,
            "formula_info": formula_info,
            "count_formula_symbols": sum(x[1] - x[0] for x in formula_info),
            "meta": document_meta,
        }
        
        if check_unknown_commands(in_context_math_data):
            continue

        in_context_math.append(in_context_math_data)

    return in_context_math

def process_one_shard(shard_i):
    
    zst_file_output = f"ext_data/arXiv_{shard_i:03}.jsonl.zst"
    formulas_file_name = f'data/formulas_{shard_i:03}.jsonl'
    # if os.path.exists(formulas_file_name):
    #     print(f"File shard {shard_i} already exsits:", formulas_file_name)
    #     return

    if not os.path.exists(zst_file_output):
        url = f'https://huggingface.co/datasets/EleutherAI/proof-pile-2/resolve/main/arxiv/train/arXiv_{shard_i:03}.jsonl.zst'
        r = requests.get(url, allow_redirects=True)
        if r.status_code != 200:
            print("Failed to download shard:", shard_i, r.status_code, r.content)
            return

        with open(zst_file_output, 'wb') as f:
            f.write(r.content)

    arxiv_jsonl_filename = decomptress_zst(zst_file_output)

    arxiv_dataset_samples = []

    with open(arxiv_jsonl_filename, "r") as json_file:
        for json_line in json_file.readlines():
            arxiv_dataset_samples.append(json.loads(json_line))


    arxiv_sentences = []

    for sample in tqdm(arxiv_dataset_samples, desc=f"Processing arxiv samples {shard_i}"):
        sample_meta = sample['meta']
        if int(sample_meta['yymm'][:2]) < 19:
            continue

        for line in sample['text'].split('\n'):
            if "$$" in line:
                continue
            if line.strip():
                for sentence in line.split('. '):
                    arxiv_sentences.append({
                        "sentence": sentence,
                        "meta": sample_meta,
                    })

    in_context_math = process_arxiv_data(arxiv_sentences);
    
    df = pd.DataFrame(in_context_math)
    df['count_formula_symbols_strates'] = df['count_formula_symbols'].apply(make_strates)
    count_formula_symbols_strates_value_counts = df['count_formula_symbols_strates'].value_counts()
    stratified_df = df.groupby('count_formula_symbols_strates', group_keys=False).apply(lambda x: x.sample(count_formula_symbols_strates_value_counts.min()))
    
    print("stratified_df", len(stratified_df))
    assert len(stratified_df) > 50000
    
    stratified_df.to_json(formulas_file_name, orient='records', lines=True)

if __name__ == "__main__":

    os.makedirs('data/', exist_ok=True)
    os.makedirs('ext_data/', exist_ok=True)
    
    with Pool(processes=10) as p:
        total_shards = 100
        r = list(tqdm(p.imap(process_one_shard, range(total_shards)), total=total_shards))
