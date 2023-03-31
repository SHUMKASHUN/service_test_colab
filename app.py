from flask import Flask, request, make_response, jsonify
from flask import render_template
from flask_cors import CORS
# from lmflow.datasets.dataset import Dataset
# from lmflow.pipeline.auto_pipeline import AutoPipeline
# from lmflow.models.auto_model import AutoModel
# from lmflow.args import ModelArguments, DatasetArguments, AutoArguments
# from lmflow.models.hf_decoder_model import HFDecoderModel
# import torch.distributed as dist
# from transformers import HfArgumentParser
# import io
import json
# import torch
# import os

app = Flask(__name__)
CORS(app)
# ds_config_path = "../examples/ds_config.json"
# with open (ds_config_path, "r") as f:
#     ds_config = json.load(f)

# model_name = 'gpt2'
# lora_path = '../output_models/instruction_ckpt/llama7b-lora/'
# model_args = ModelArguments(model_name_or_path=model_name)

# local_rank = int(os.getenv("LOCAL_RANK", "0"))
# world_size = int(os.getenv("WORLD_SIZE", "1"))
# torch.cuda.set_device(local_rank)
# model = AutoModel.get_model(model_args, tune_strategy='none', ds_config=ds_config)
import time
@app.route('/predict',methods = ['POST'])
def predict():
    if(request.method == "POST"):
        try:
            user_input = request.get_json()["Input"]
            conversation = request.get_json()["History"]
            print(conversation)
            print(user_input)
            text_out = "hey hey"
        except:
            text_out = "There is something wrong, please query again"
        # user_input is a string
        # inputs = model.encode(user_input, return_tensors="pt").to(device=local_rank)
        # outputs = model.inference(inputs, max_new_tokens=100,temperature=0.0, do_sample=False)
        # text_out = model.decode(outputs[0], skip_special_tokens=True)
        # prompt_length = len(model.decode(inputs[0], skip_special_tokens=True,))
        # text_out = text_out[prompt_length:].strip("\n")
        # print(text_out)
    else:
        text_out = "pending"
    return text_out



@app.route('/',methods = ['GET'])
def login():

    return render_template('index.html')


app.run(port = 8888, debug = True)
