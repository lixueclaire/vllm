from vllm import LLM, SamplingParams
import time
import os
# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# Create an LLM.
# model_path = "/mnt/nas/tao/models/Llama-2-70B-GPTQ"
model_path = "/root/models/Llama-2-70B-GPTQ"
start_time = time.perf_counter()
#llm = LLM(model=model_path)
llm = LLM(model=model_path, enforce_eager=True, tensor_parallel_size=2, instance_num=1)
#llm = LLM(model=model_path, enforce_eager=True, tensor_parallel_size=2)
end_time = time.perf_counter()
total_time = end_time - start_time

# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

print(f"LLM generation took {total_time:.2f} seconds.")
