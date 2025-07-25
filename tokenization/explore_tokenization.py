from transformers import T5Tokenizer, AutoTokenizer

if __name__ == '__main__':

    path_corrector = "AAAI2025/MathSpeech_Ablation_Study_LaTeX_translator_T5_small" # All T5 models were trained using the same tokenizer.

    math_speech_tokenizer = T5Tokenizer.from_pretrained(path_corrector)
    qwen_math_tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-Math-7B')
    phi_mini_tokenizer = AutoTokenizer.from_pretrained('microsoft/Phi-4-mini-flash-reasoning')

    all_tokenizers = [
        math_speech_tokenizer,
        qwen_math_tokenizer,
        phi_mini_tokenizer,
    ]

    check_symbols_in_tokenizer = '{}_^\\<>~'
    tokenization_examples = [ '\\lim_{x}{\\frac{1}{x}}', '\\lim_{ x } { \\frac{ 1 }{ x }}' ]

    for tokenizer in all_tokenizers:
        print("\n\n tokenizer", type(tokenizer))


        for symbol in check_symbols_in_tokenizer:
            token_exists = tokenizer.convert_tokens_to_ids(symbol) != tokenizer.unk_token_id
            token_exists_emoji = '⚠️' if not token_exists else '✅'
            print(f'symbol {symbol}', token_exists_emoji, "decoded", tokenizer.decode(tokenizer.convert_tokens_to_ids(symbol)))

        for sample in tokenization_examples:
            tokens = tokenizer.encode(sample)
            sum_unknown = sum(1 for token in tokens if token == tokenizer.unk_token_id)
            print(f'sample {sample}')
            print(f'tokens count {len(tokens)}')
            print(f'unknown tokens count {sum_unknown}', '⚠️' if sum_unknown > 0 else '✅')
            print(f'tokens {tokens}')
            print(f'decoded {tokenizer.batch_decode(tokens)}')


# Script output
# Note that MathSpeech tokenizer has following extra special tokens
#         32100: AddedToken("\", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
#         32101: AddedToken("{", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
#         32102: AddedToken("}", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
#         32103: AddedToken("^", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
#         32104: AddedToken("<", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
#         32105: AddedToken("~", rstrip=False, lstrip=False, single_word=False, normalized=False, special=True),
"""
 tokenizer <class 'transformers.models.t5.tokenization_t5.T5Tokenizer'>
symbol { ✅ decoded {
symbol } ✅ decoded }
symbol _ ✅ decoded _
symbol ^ ✅ decoded ^
symbol \ ✅ decoded \
symbol < ✅ decoded <
symbol > ✅ decoded >
symbol ~ ✅ decoded ~
sample \lim_{x}{\frac{1}{x}}
tokens count 21
unknown tokens count 0 ✅
tokens [32100, 3, 4941, 834, 32101, 3, 226, 32102, 32101, 32100, 3, 9880, 32101, 209, 32102, 32101, 3, 226, 32102, 32102, 1]
decoded ['\\', '', 'lim', '_', '{', '', 'x', '}', '{', '\\', '', 'frac', '{', '1', '}', '{', '', 'x', '}', '}', '</s>']
sample \lim_{ x } { \frac{ 1 }{ x }}
tokens count 21
unknown tokens count 0 ✅
tokens [32100, 3, 4941, 834, 32101, 3, 226, 32102, 32101, 32100, 3, 9880, 32101, 209, 32102, 32101, 3, 226, 32102, 32102, 1]
decoded ['\\', '', 'lim', '_', '{', '', 'x', '}', '{', '\\', '', 'frac', '{', '1', '}', '{', '', 'x', '}', '}', '</s>']


 tokenizer <class 'transformers.models.qwen2.tokenization_qwen2_fast.Qwen2TokenizerFast'>
symbol { ✅ decoded {
symbol } ✅ decoded }
symbol _ ✅ decoded _
symbol ^ ✅ decoded ^
symbol \ ✅ decoded \
symbol < ✅ decoded <
symbol > ✅ decoded >
symbol ~ ✅ decoded ~
sample \lim_{x}{\frac{1}{x}}
tokens count 12
unknown tokens count 0 ✅
tokens [59, 4659, 15159, 87, 15170, 59, 37018, 90, 16, 15170, 87, 3417]
decoded ['\\', 'lim', '_{', 'x', '}{', '\\', 'frac', '{', '1', '}{', 'x', '}}']
sample \lim_{ x } { \frac{ 1 }{ x }}
tokens count 15
unknown tokens count 0 ✅
tokens [59, 4659, 15159, 856, 335, 314, 1124, 37018, 90, 220, 16, 335, 90, 856, 3869]
decoded ['\\', 'lim', '_{', ' x', ' }', ' {', ' \\', 'frac', '{', ' ', '1', ' }', '{', ' x', ' }}']


 tokenizer <class 'transformers.models.gpt2.tokenization_gpt2_fast.GPT2TokenizerFast'>
symbol { ✅ decoded {
symbol } ✅ decoded }
symbol _ ✅ decoded _
symbol ^ ✅ decoded ^
symbol \ ✅ decoded \
symbol < ✅ decoded <
symbol > ✅ decoded >
symbol ~ ✅ decoded ~
sample \lim_{x}{\frac{1}{x}}
tokens count 12
unknown tokens count 0 ✅
tokens [59, 5406, 22305, 87, 29124, 59, 63757, 90, 16, 29124, 87, 6478]
decoded ['\\', 'lim', '_{', 'x', '}{', '\\', 'frac', '{', '1', '}{', 'x', '}}']
sample \lim_{ x } { \frac{ 1 }{ x }}
tokens count 15
unknown tokens count 0 ✅
tokens [59, 5406, 22305, 1215, 388, 354, 2381, 63757, 90, 220, 16, 388, 90, 1215, 10425]
decoded ['\\', 'lim', '_{', ' x', ' }', ' {', ' \\', 'frac', '{', ' ', '1', ' }', '{', ' x', ' }}']
"""