import argparse
import json
import time
import torch
import vllm
from vllm import LLM, SamplingParams

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--input-len", type=int, required=True)
    parser.add_argument("--output-len", type=int, required=True)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--num-prompts", type=int, required=True)
    parser.add_argument("--gpu-mem-util", type=float, default=0.9)
    parser.add_argument("--max-model-len", type=int, required=False)
    parser.add_argument("--max-num-batched-tokens", type=int, required=False)
    parser.add_argument("--result-file", type=str, required=True)
    args = parser.parse_args()

    print(f"Initializing LLM offline engine for {args.model}...")
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=args.gpu_mem_util,
        max_num_seqs=args.concurrency,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        enforce_eager=False,
        disable_log_stats=False,
    )

    # We use prompt_token_ids to bypass tokenizer overhead entirely,
    # supplying the exact input length requested.
    dummy_prompt_token_ids = [[1] * args.input_len for _ in range(args.num_prompts)]
    
    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        ignore_eos=True,
        max_tokens=args.output_len,
    )

    vllm_version = tuple(map(int, vllm.__version__.split('.')[:2]))

    print("Warming up engine (1 prompt)...")
    if vllm_version >= (0, 15):
        llm.generate(prompts=[{"prompt_token_ids": [1] * args.input_len}], sampling_params=sampling_params, use_tqdm=False)
    else:
        llm.generate(prompt_token_ids=[[1] * args.input_len], sampling_params=sampling_params, use_tqdm=False)

    print("Starting profiled generation...")
    # This turns on the CUDA profiler (which nsys --capture-range=cudaProfilerApi waits for)
    torch.cuda.profiler.start()
    
    start_time = time.perf_counter()
    if vllm_version >= (0, 15):
        outputs = llm.generate(prompts=[{"prompt_token_ids": pid} for pid in dummy_prompt_token_ids], sampling_params=sampling_params, use_tqdm=True)
    else:
        outputs = llm.generate(prompt_token_ids=dummy_prompt_token_ids, sampling_params=sampling_params, use_tqdm=True)
    end_time = time.perf_counter()
    
    torch.cuda.profiler.stop()

    print(f"Generation completed in {end_time - start_time:.2f}s")
    
    # Save a minimal JSON so artifacts still upload something
    result = {
        "model": args.model,
        "input_len": args.input_len,
        "output_len": args.output_len,
        "concurrency": args.concurrency,
        "num_prompts": args.num_prompts,
        "duration_s": end_time - start_time,
        "throughput_tok_s": (args.num_prompts * args.output_len) / (end_time - start_time)
    }
    with open(args.result_file, "w") as f:
        json.dump(result, f, indent=2)

if __name__ == "__main__":
    main()
