import torch
import pickle
import jittor as jt

pytorch_weights = torch.load("CamoFormer-trained", map_location="cpu")

if "state_dict" in pytorch_weights:
    pytorch_weights = pytorch_weights["state_dict"]

jittor_weights = {}
for key, value in pytorch_weights.items():
    if value is not None:
        jittor_weights[key] = jt.array(value.numpy())  # 转换为 jittor.Var
    else:
        print(f"Skipping parameter {key} as it is None.")

with open("camoformer_pvtv2b4.pkl", "wb") as f:
    pickle.dump(jittor_weights, f)
    
# with open("camoformer_pvtv2b4.pkl", "rb") as f:
#     jittor_weights = pickle.load(f)
    
# if "encoder.block1.0.attn.q.bias" in jittor_weights:
#     print("Parameter found in weights.")
#     print(f"Value: {jittor_weights['encoder.block1.0.attn.q.bias']}")
# else:
#     print("Parameter not found in weights.")

# for key, value in jittor_weights.items():
#     # print("key: ",key)
#     # print("value: ",value)
#     if key == "encoder.block1.0.attn.q.bias":
#         print(value)
#     if value is None:
#         print(f"Parameter {key} is None!")
