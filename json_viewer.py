import ijson

filename = "./data/comments.json"

output_file = "./data/first_10000_chars.json"

with open(filename, "r") as f:
    snippet = f.read(10000)

with open(output_file, "w") as f:
    f.write(snippet)