
####################################
####################################
#####nfer infer with mellum_4b #####



from transformers import AutoTokenizer, AutoModelForCausalLM

# Example prompt
example = """


def quick_sort(arr):
"""

# Load tokenizer and model
model_name = "JetBrains/Mellum-4b-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Tokenize input
inputs = tokenizer(example, return_tensors='pt', return_token_type_ids=False)
input_ids = inputs["input_ids"]

# Generate continuation
output_ids = model.generate(
    input_ids=input_ids,
    max_new_tokens=256,
    do_sample=True,  # Optional: adds randomness for more diverse outputs
    temperature=0.7,  # Optional: controls creativity vs. determinism
    top_p=0.95,       # Optional: nucleus sampling
    pad_token_id=tokenizer.eos_token_id  # Prevents warning if EOS is not defined
)

# Decode and print
generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

# Split context and prediction
generated_lines = generated_text.splitlines()
context_lines = example.strip().splitlines()
prediction_lines = generated_lines[len(context_lines):]

print("### Context")
print(example.strip())
print("### Prediction")
print("\n".join(prediction_lines))


print('done')
