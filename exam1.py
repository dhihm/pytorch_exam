import transformers
from transformers import pipeline

# Load the GPT-2 Neo Model
generator = pipeline('text-generation', model='EleutherAI/gpt-neo-2.7B')

# Generate text
text = generator("Once upon a time", max_length=100)[0]['generated_text']
print(text)

# tokenize the text
tokens = generator.tokenizer(text)
print(tokens)

# convert the tokens to ids
ids = generator.tokenizer.convert_tokens_to_ids(tokens)
print(ids)

# run 
print(generator.model.generate(ids, do_sample=True, max_length=100, pad_token_id=generator.tokenizer.eos_token_id))


