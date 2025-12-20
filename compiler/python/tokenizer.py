from transformers import AutoTokenizer


def load_tokenizer(model_name="gpt2"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer

if __name__ == "__main__":
    # Example usage
    tokenizer = load_tokenizer("gpt2")

    text = "Using a Transformer network is simple"
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)

    print("Original text:", text)
    print("Token IDs:", encoded)
    print("Decoded text:", decoded)

tokenizer = load_tokenizer("gpt2")
vocab_size = tokenizer.vocab_size
print(f"The vocabulary size for GPT-2 is: {vocab_size}")
