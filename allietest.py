# data exploration

# use ijson parser for large json file
import ijson

# json file is structured as a dictionary - go through key value pairs
count = 0
with open("comments.json", "rb") as f:
    for key, value in ijson.kvitems(f, ""):
        count += 1

print("Total items:", count) # 12704751 items