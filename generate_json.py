import json

def deserialize_file(file_path):
  with open(file_path, 'r+') as json_file:
    json_list = list(json_file)
  full_dict = {}
  f = open('test.jsonl', 'w')
  i = 0
  for json_str in json_list:
    result = json.loads(json_str)
    result['context'] += 'the the the'
    json.dump(result, f)
    full_dict[i] = result
    i += 1
  f.close()
  return full_dict

def serialize_file(full_dict):
  f = open('test.json', 'w')
  json.dump(full_dict, f)
  f.close()


full_dict = deserialize_file('./squad_train.jsonl')
# serialize_file(full_dict)