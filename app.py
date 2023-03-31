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
list_a = ["123","234","345"]
type_list = ["1","2","1"]
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
@app.route('/predict',methods = ['POST','GET'])
def predict():
    print(request)
    return "Hello2"

@app.route('/',methods = ['POST','GET'])
def login():
    user_input = request.form.get('user_input')
    print(request.form)
    if user_input != None:
        user_input = user_input.strip("\r").strip("\n").strip(" ")
        print(len(user_input.replace(u'\xa0', u'').strip(" ")))
    if(request.method == "POST"):
        text_out = user_input
        list_a[2] = "cap"
        # user_input is a string
        # inputs = model.encode(user_input, return_tensors="pt").to(device=local_rank)
        # outputs = model.inference(inputs,min_length=5, max_length=100,temperature=0.0, do_sample=False)
        # text_out = model.decode(outputs[0], skip_special_tokens=True)
        # prompt_length = len(model.decode(inputs[0], skip_special_tokens=True,))
        # text_out = text_out[prompt_length:].strip("\n")
        # print(text_out)
    else:
        text_out = "pending"
    print(list_a)

    return render_template('index.html',msg = text_out, demo_list = list_a, type_list = type_list, title_variable = "hhh")


app.run(port = 8888, debug = True)
