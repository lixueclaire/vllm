from vllm import LLM, SamplingParams
import time
import gc

# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

# Sampling parameters.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# Models.
llama_7b = "/home/lixue_models/Llama-2-7b-hf"
llama_7b_chat = "/home/lixue_models/Llama-2-7b-chat-hf"
llama3_8b = "/home/lixue_models/Llama-3-8B-Instruct"
llama_13b = "/home/lixue_models/Llama-2-13b-chat-hf"
qwen_7b = "/home/lixue_models/Qwen-7B-Chat"
tensor_parallel_size = 2

def generate_text(llm, prompts, sampling_params):
    # Generate texts.
    outputs = llm.generate(prompts, sampling_params)
    # Print the generated texts.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


# Init LLM
# Mark the start time.
start_time = time.perf_counter()
# Create an LLM instance.
llm = LLM(
    model=llama_7b, 
    tensor_parallel_size=tensor_parallel_size, 
    disable_custom_all_reduce=True,
    enforce_eager=True, 
    trust_remote_code=True)
# Mark the end time and calculate total time elapsed.
end_time = time.perf_counter()
total_time = end_time - start_time
print(f"LLM start took {total_time:.2f} seconds.")
# Test.
outputs = generate_text(llm, prompts, sampling_params)

# Swap LLM
start_time = time.perf_counter()
llm.swap(
    model=llama_7b_chat, 
    tensor_parallel_size=tensor_parallel_size, 
    disable_custom_all_reduce=True,
    enforce_eager=True, 
    trust_remote_code=True)
end_time = time.perf_counter()
total_time = end_time - start_time
print(f"LLM swap took {total_time:.2f} seconds.")
# Test.
outputs = generate_text(llm, prompts, sampling_params)

# Swap LLM
start_time = time.perf_counter()
llm.swap(model=llama3_8b, tensor_parallel_size=tensor_parallel_size, enforce_eager=True, trust_remote_code=True)
end_time = time.perf_counter()
total_time = end_time - start_time
print(f"LLM swap took {total_time:.2f} seconds.")
# Test.
outputs = generate_text(llm, prompts, sampling_params)