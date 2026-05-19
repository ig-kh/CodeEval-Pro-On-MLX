import xgrammar as xgr
import mlx.core as mx
from mlx_lm import generate
from mlx_lm.sample_utils import make_sampler, make_repetition_penalty

class XGrammarLogitsProcessor:
    def __init__(self, hf_tokenizer, grammar):
        self.hf_tokenizer = hf_tokenizer
        self.eos_token_id = hf_tokenizer.eos_token_id
        self.vocab_size_hf = len(hf_tokenizer)
        self.matcher = xgr.GrammarMatcher(grammar)
        self.bitmask = xgr.allocate_token_bitmask(1, self.vocab_size_hf)
        self.terminated = False

    def __call__(self, input_ids: mx.array, logits: mx.array) -> mx.array:
        # If already terminated, only allow EOS token
        if self.terminated:
            vocab_mlx = logits.shape[-1]
            mask = mx.zeros((vocab_mlx,), dtype=mx.bool_)
            if self.eos_token_id < vocab_mlx:
                mask[self.eos_token_id] = True
            logits_1d = logits[0]
            logits_1d = mx.where(mask, logits_1d, -float('inf'))
            return logits_1d.reshape(1, -1)

        # Accept the last generated token (skip the initial prompt)
        if input_ids.size > 1:
            last_token = int(input_ids[-1].item())
            self.matcher.accept_token(last_token)

        # Try to get the next token mask; catch termination error
        try:
            self.matcher.fill_next_token_bitmask(self.bitmask, 0)
        except RuntimeError as e:
            if "GrammarMatcher has terminated" in str(e):
                self.terminated = True
                vocab_mlx = logits.shape[-1]
                mask = mx.zeros((vocab_mlx,), dtype=mx.bool_)
                if self.eos_token_id < vocab_mlx:
                    mask[self.eos_token_id] = True
                logits_1d = logits[0]
                logits_1d = mx.where(mask, logits_1d, -float('inf'))
                return logits_1d.reshape(1, -1)
            else:
                raise

        # Decode the bitmask to allowed token IDs
        bitmask_np = self.bitmask.numpy().flatten()
        allowed_ids_hf = []
        for block_idx, block in enumerate(bitmask_np):
            block_val = int(block)
            for bit in range(32):
                token_id = block_idx * 32 + bit
                if token_id >= self.vocab_size_hf:
                    break
                if (block_val >> bit) & 1:
                    allowed_ids_hf.append(token_id)

        vocab_mlx = logits.shape[-1]
        mask = mx.zeros((vocab_mlx,), dtype=mx.bool_)
        for tid in allowed_ids_hf:
            if tid < vocab_mlx:
                mask[tid] = True

        logits_1d = logits[0]
        logits_1d = mx.where(mask, logits_1d, -float('inf'))
        return logits_1d.reshape(1, -1)

def load_grammar(grammar_path: str, mlx_tokenizer):
    """Compile grammar using the underlying Hugging Face tokenizer."""
    hf_tokenizer = mlx_tokenizer._tokenizer
    tokenizer_info = xgr.TokenizerInfo.from_huggingface(hf_tokenizer)
    compiler = xgr.GrammarCompiler(tokenizer_info)
    with open(grammar_path, 'r') as f:
        grammar_content = f.read()
    if grammar_path.endswith('.json'):
        grammar = xgr.Grammar.from_json_schema(grammar_content)
    else:
        grammar = xgr.Grammar.from_ebnf(grammar_content)
    return compiler.compile_grammar(grammar)

def generate_with_grammar(model, mlx_tokenizer, prompt, grammar,
                          max_tokens=2048, temperature=0.2, top_p=0.95,
                          repetition_penalty=1.0):
    """Constrained generation using XGrammar with a custom logits processor."""
    hf_tokenizer = mlx_tokenizer._tokenizer
    processor = XGrammarLogitsProcessor(hf_tokenizer, grammar)
    logits_processors = [processor]

    if repetition_penalty != 1.0:
        rep_processor = make_repetition_penalty(repetition_penalty)
        logits_processors.append(rep_processor)

    sampler = make_sampler(temp=temperature, top_p=top_p)

    response = generate(
        model,
        mlx_tokenizer,
        prompt=prompt,
        sampler=sampler,
        logits_processors=logits_processors,
        max_tokens=max_tokens,
        verbose=False
    )
    return response