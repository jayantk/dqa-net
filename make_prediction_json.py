import json
import sys

tf_json = sys.argv[1]
question_json = sys.argv[2]


id_map = {}
with open(question_json, 'r') as f:
    for line in f:
        j = json.loads(line)
        diagram_id = j["diagram_id"].split('.')[0]
        
        if not diagram_id in id_map:
            id_map[diagram_id] = []
        id_map[diagram_id].append(j["id"])

with open(tf_json, 'r') as f:
    predictions = json.load(f)
    yp = predictions["values"]["yp"]
    ids = predictions["ids"]

    for i in range(len(ids)):
        (diag_id, ind) = ids[i]

        if (len(id_map[diag_id]) > ind):
            fw_id = id_map[diag_id][ind]
            j = {'questionId' : fw_id, 'answerProbs' : yp[i], 'guess' : False}
            print(json.dumps(j))
        else:
            print("unknown question:", ids[i], file=sys.stderr)
