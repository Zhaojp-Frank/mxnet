import json

with open("profile_output.json", "r") as infile:
    js_p = json.load(infile)

with open("resnet.json", "r") as infile:
    js_d = json.load(infile);

i = 0
_events = js_p["traceEvents"]
events = []
for event in _events:
    if event['ph'] != 'M' and event['pid'] == 24:
        events.append(event)

for j, node in enumerate(js_d["nodes"]):
    time = 0
    #print(i,j)
    if node["op"] != "null":
        if events[i]["name"] != node["op"]:
            print("At ts =", events[i]["ts"], "i =", i, "op and operator does not match")
            print("Profiler =", events[i]["name"])
            print("Expected =", node["op"])
            break;
        time = events[i+1]["ts"] - events[i]["ts"]
        while j != len(js_d["nodes"])-1 and events[i+2]["ts"] < events[i+1]["ts"]:
            i+=2
        i+=2
    node["time"] = time

with open("resnet_time.json", "w") as outfile:
    json.dump(js_d, outfile, sort_keys=True, indent=4)
