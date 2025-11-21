import ijson

filename = "comments.json"

output_file = "first_10000_chars.json"

with open(filename, "r") as f:
    snippet = f.read(100000)

with open(output_file, "w") as f:
    f.write(snippet)