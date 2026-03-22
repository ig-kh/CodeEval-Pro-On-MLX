import torch
import itertools
from pathlib import Path
from dataclasses import dataclass

from mlx_lm import load, generate
from mlx_lm.sample_utils import (
    make_sampler,
    make_logits_processors,
    make_repetition_penalty,
)
from mlx_lm.models.cache import make_prompt_cache
from tqdm.auto import tqdm
from typing import Any, Literal, Dict, Mapping, Iterable, Sequence, TypeVar, cast
from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers import (
    HfArgumentParser,
    GenerationConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
)

from eval.prompt_template import PROMPT_WRAPPER
from eval.utils import (
    read_jsonl,
    write_jsonl,
    get_humaneval_raw_problems,
    get_mbpp_raw_problems,
    get_mbpp_pro_raw_problems,
    get_humaneval_pro_raw_problems,
    map_humaneval_problem,
    map_mbpp_problem,
    map_humaneval_pro_problem,
    map_mbpp_pro_problem,
    map_humaneval_pro_problem_cot,
    map_mbpp_pro_problem_cot,
    map_humaneval_pro_problem_1shot,
    map_mbpp_pro_problem_1shot,
    get_bigcodebench_lite_pro_problems,
    map_bigcodebench_lite_pro_problem,
)

_T = TypeVar("_T")
CUDA_NUM = torch.cuda.device_count()
EOS = [
    "<|endoftext|>",
    "<|endofmask|>",
    "</s>",
    "<|EOT|>",
    # "\nif __name__",
    # "\ndef main(",
    # "\nprint(",
]

DATASET_MAPPING = {
    "humaneval": (get_humaneval_raw_problems, map_humaneval_problem),
    "mbpp": (get_mbpp_raw_problems, map_mbpp_problem),
    "humaneval_pro": (get_humaneval_pro_raw_problems, map_humaneval_pro_problem),
    "mbpp_pro": (get_mbpp_pro_raw_problems, map_mbpp_pro_problem),
    "humaneval_pro_cot": (
        get_humaneval_pro_raw_problems,
        map_humaneval_pro_problem_cot,
    ),
    "mbpp_pro_cot": (get_mbpp_pro_raw_problems, map_mbpp_pro_problem_cot),
    "humaneval_pro_1shot": (
        get_humaneval_pro_raw_problems,
        map_humaneval_pro_problem_1shot,
    ),
    "mbpp_pro_1shot": (get_mbpp_pro_raw_problems, map_mbpp_pro_problem_1shot),
    "bigcodebench_lite_pro": (
        get_bigcodebench_lite_pro_problems,
        map_bigcodebench_lite_pro_problem,
    ),
}


@dataclass(frozen=True)
class Args:
    dataset: Literal["humaneval", "mbpp", "humaneval_pro", "mbpp_pro"]
    save_path: str
    n_batches: int
    n_problems_per_batch: int
    n_samples_per_problem: int
    max_new_tokens: int
    temperature: float
    top_p: float
    do_sample: bool
    model_name_or_path: str | None = None
    use_flash_attention: bool = False
    # MLX-specific
    use_mlx: bool = False
    repetition_penalty: float = 1.0
    lazy: bool = True
    max_kv_size: int = 4096


@dataclass
class MLXGenerationConfig:
    max_new_tokens: int
    temperature: float = 0.6
    top_p: float = 0.9
    repetition_penalty: float = 1.0
    do_sample: bool = True
    num_return_sequences: int = 1


@dataclass(frozen=True)
class ModelContext:
    model: Any
    tokenizer: Any = None
    backend: Literal["hf", "mlx"] = "hf"

    def complete(
        self, config: GenerationConfig | MLXGenerationConfig, prompts: list[str]
    ) -> Dict:
        if self.backend == "mlx":
            if not isinstance(config, MLXGenerationConfig):
                raise ValueError("MLX backend requires MLXGenerationConfig")

            sampler = make_sampler(temp=config.temperature, top_p=config.top_p)
            logits_processors = []
            if config.repetition_penalty != 1.0:
                logits_processors.append(
                    make_repetition_penalty(config.repetition_penalty)
                )

            prompt_cache = make_prompt_cache(self.model)

            output_strings = []
            for prompt in prompts:
                samples = []
                for _ in range(config.num_return_sequences):
                    response = generate(
                        self.model,
                        self.tokenizer,
                        prompt=prompt,
                        sampler=sampler,
                        logits_processors=(
                            logits_processors if logits_processors else None
                        ),
                        max_tokens=config.max_new_tokens,
                        prompt_cache=prompt_cache,
                        verbose=False,
                    )
                    samples.append(response)
                output_strings.append(samples)
            return {"decoded_outputs": output_strings}

        else:  # Hugging Face backend
            if self.tokenizer is None:
                raise ValueError("HF backend requires tokenizer")
            self.tokenizer.pad_token = self.tokenizer.eos_token
            input_ids = self.tokenizer(prompts, return_tensors="pt", padding=True)
            input_len = input_ids["input_ids"].shape[-1]
            input_ids = input_ids.to(self.model.device)
            output_ids = self.model.generate(**input_ids, generation_config=config)
            output_ids = output_ids[:, input_len:]
            output_strings = self.tokenizer.batch_decode(
                output_ids, skip_special_tokens=True
            )
            return {"decoded_outputs": output_strings}


def chunked(seq: Sequence[_T], n: int) -> Iterable[Sequence[_T]]:
    """Yield successive n-sized chunks from seq."""
    return (seq[i : i + n] for i in range(0, len(seq), n))


def main():
    parser = HfArgumentParser(Args)
    args = cast(Args, parser.parse_args_into_dataclasses())[0]

    raw_problem_fn, map_problem_fn = DATASET_MAPPING[args.dataset]
    raw_problems = raw_problem_fn()
    problems = list(map(map_problem_fn, raw_problems))

    if args.use_mlx:
        # MLX backend
        generation_config = MLXGenerationConfig(
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p if args.do_sample else 1.0,
            repetition_penalty=args.repetition_penalty,
            do_sample=args.do_sample,
            num_return_sequences=args.n_samples_per_problem,
            # max_kv_size=args.max_kv_size
        )
        model, tokenizer = load(
            args.model_name_or_path,
            lazy=args.lazy,
        )
        backend = "mlx"
    else:
        # Hugging Face backend
        other_kwargs = {"device_map": "auto"}
        if args.use_flash_attention:
            other_kwargs["use_flash_attention_2"] = True

        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path, trust_remote_code=True, **other_kwargs
        )
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path,
            use_fast=True,
            trust_remote_code=True,
            padding_side="left",
        )
        generation_config = GenerationConfig(
            do_sample=args.do_sample,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            num_return_sequences=args.n_samples_per_problem,
            top_p=args.top_p,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        backend = "hf"

    state = ModelContext(model, tokenizer, backend=backend)

    problems_chunked = list(chunked(problems, args.n_problems_per_batch))
    total_batches = (
        min(len(problems_chunked), args.n_batches)
        if args.n_batches > 0
        else len(problems_chunked)
    )
    problems_chunked = problems_chunked[:total_batches]

    samples = []
    Path(args.save_path).write_text("")

    for batch_idx, batch_problems in enumerate(
        tqdm(problems_chunked, total=len(problems_chunked))
    ):
        task_ids = [problem["id"] for problem in batch_problems]
        prompts = [
            PROMPT_WRAPPER.format(
                instruction=problem["instruction"],
                response=problem.get("response_prefix", ""),
            )
            for problem in batch_problems
        ]

        print("PROMPT")
        print(prompts[-1])
        # all_prompts = prompts * args.n_samples_per_problem
        all_task_ids = task_ids * args.n_samples_per_problem
        response = state.complete(generation_config, prompts)
        completions = response["decoded_outputs"]
        print("COMPLETION")
        print(completions[-1])
        if len(completions[-1]) == 1:
            completions = [c[0] for c in completions]
            samples += [
                dict(
                    task_id=task_id,
                    completion=completion[
                        : (
                            index
                            if (index := completion.find("```")) != -1
                            else len(completion)
                        )
                    ],
                )
                for task_id, completion in zip(all_task_ids, completions)
            ]
        else:
            samples += [
                dict(
                    task_id=task_id,
                    completion=[
                        completion[
                            : (
                                index
                                if (index := completion.find("```")) != -1
                                else len(completion)
                            )
                        ]
                        for completion in completion_batch
                    ],
                )
                for task_id, completion_batch in zip(all_task_ids, completions)
            ]
        # print(samples)
        write_jsonl(args.save_path, samples)


if __name__ == "__main__":
    main()
