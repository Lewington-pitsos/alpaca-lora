from datasets import load_dataset

data = load_dataset("json", data_files="alpaca_data_cleaned.json")

print(len(data))