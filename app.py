import os
directory = os.getcwd()
os.chdir("D:/ask-anything/video_chat_with_stablelm/")
import json
import numpy as np
import random
import torch
import torchvision.transforms as transforms
from PIL import Image
from models.tag2text import tag2text_caption
from util import *
import gradio as gr
#from stablelm import *
#from chatbot import *
import argparse
from load_internvideo import *
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from models.grit_model import DenseCaptioning
import time

parser = argparse.ArgumentParser(description='vid 2 text')
parser.add_argument('--desc', dest='desc', type=str, default=None, help="Description")
parser.add_argument('--path', dest='path', type=str, default='./in_full.mp4', help="INput path with filename")
parser.add_argument('--json', dest='opath', type=str, default='./vid_caption.json', help="Output path with filename.json")

args = parser.parse_args()



image_size = 224
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
transform = transforms.Compose([transforms.ToPILImage(),transforms.Resize((image_size, image_size)),transforms.ToTensor(),normalize])

if device != 'cpu':
    print("[SETUP] Running video understanding on GPU")
else:
    print("[SETUP] Running video understanding on CPU")
# define model
model = tag2text_caption(pretrained="pretrained_models/tag2text_swin_14m.pth", image_size=image_size, vit='swin_b' )
model.eval()
model = model.to(device)
print("[INFO] initialize caption model success!")

# action recognition
intern_action = load_intern_action("cpu")
trans_action = transform_action()
topil =  T.ToPILImage()
print("[INFO] initialize InternVideo model success!")

dense_caption_model = DenseCaptioning(device)
dense_caption_model.initialize_model()
print("[INFO] initialize dense caption model success!")
print("[INFO] Loading models done")
#bot = ConversationBot()

def inference(video_path, input_tag='', output='./vid_caption.json', optimize=1, progress=gr.Progress()):
    print("Watching video...")
    data = loadvideo_decord_origin(video_path)
    progress(0.2, desc="Loading Videos")
    print("Step 1/4")
    # InternVideo
    action_index = np.linspace(0, len(data)-1, 8).astype(int)
    action_tensor = []
    tmp,tmpa = [],[]
    for i,img in enumerate(data):
        tmp.append(transform(img).to(device).unsqueeze(0))
        if i in action_index and not optimize:
            tmpa.append(topil(img))
    prediction=''
    action_tensor = trans_action(tmpa)
    TC, H, W = action_tensor.shape
    action_tensor = action_tensor.reshape(1, TC//3, 3, H, W).permute(0, 2, 1, 3, 4).to("cpu")
    with torch.no_grad():
        prediction = intern_action(action_tensor)
        prediction = F.softmax(prediction, dim=1).flatten()
        prediction = kinetics_classnames[str(int(prediction.argmax()))]
    print("Step 2/4")
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    
    # dense caption
    dense_caption = []
    if device == "cpu" or optimize:
        dense_index = np.arange(0, len(data)-1, 15)
    else:
        dense_index = np.arange(0, len(data)-1, 5)    

    original_images = data[dense_index,:,:,::-1]
    dcs = {}
    with torch.no_grad():
        dcc=0
        dcl = len(original_images)
        for original_image in original_images:
            dense_caption.append(dense_caption_model.run_caption_tensor(original_image))
            dcc += 1
            if device == "cpu" or optimize:
                print(f"{dcc} of {dcl}")
        #dense_caption = ' '.join([f"Second {i+1} : {j}.\n" for i,j in zip(dense_index,dense_caption)])
        for i,j in zip(dense_index,dense_caption):
            key = f"{i+1}"
            value = f"\n View at {i+1} seconds: {j}.\n"
            dcs[key] = value
    print("Step 3/4")  
    
    
    del data, action_tensor, original_image, tmpa
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    
    # Video Caption
    image = torch.cat(tmp).to(device)   
    del tmp

    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    model.threshold = 0.68
    if input_tag == '' or input_tag == 'none' or input_tag == 'None':
        input_tag_list = None
    else:
        input_tag_list = []
        input_tag_list.append(input_tag.replace(',',' | '))
    with torch.no_grad():
        print("Step 4/4")
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        tag_1=[]
        tag_2=[]
        if optimize:
            caption, tag_predict = model.generate(image,tag_input = input_tag_list,max_length = 32, return_tag_predict = True)
        else:
            caption, tag_predict = model.generate(image,tag_input = input_tag_list,max_length = 50, return_tag_predict = True)
        progress(0.6, desc="Watching Videos")
        #frame_caption = ' '.join([f"Second {i+1}:{j}."+str(dcs.get(str(i+1), ""))+"\n" for i,j in enumerate(caption)])
        frame_caption = ""
        prev_caption = ""
        start_time = 0
        end_time = 0
        last_valid_dcs = ''
        for i, j in enumerate(caption):
            current_caption = f"{j}."
            current_dcs = dcs.get(f"{i+1}", "")
            if len(current_dcs) > 0:
                last_valid_dcs = current_dcs
            if current_caption == prev_caption:
                end_time = i+1
            else:
                if prev_caption:
                    frame_caption += f"Second {start_time} - {end_time}: {prev_caption}{last_valid_dcs}\n"
                start_time = i+1
                end_time = i+1
                prev_caption = current_caption
        if prev_caption:
            frame_caption += f"Second {start_time} - {end_time}: {prev_caption}{current_dcs}\n"
        total_dur = end_time
        frame_caption += f"| Total Duration: {total_dur} seconds.\n"
        if not optimize:
            if input_tag_list == None:
                tag_1 = set(tag_predict)
                tag_2 = ['none']
            else:
                _, tag_1 = model.generate(image,tag_input = None, max_length = 50, return_tag_predict = True)
                tag_2 = set(tag_predict)
        progress(0.8, desc="Understanding Videos")
        
    print("[INFO]" + video_path + " Analyzed")
    print("[VIDEO_TRANSCRIPT]")
    print(frame_caption)
    print("[TAGS1] "+ str( ','.join(tag_1) ))
    print("[TAGS2] "+ str( ','.join(tag_2) ))


    returnarray = {}
    returnarray["dense_caption"]=dcs
    returnarray["frame_caption"]=' '.join([f"Second {i+1}:{j}.\n" for i,j in enumerate(caption)])
    returnarray["prompt"]=frame_caption
    returnarray["tags"]=str( ' | '.join(tag_1) + ' | '.join(tag_2))
    returnarray["video_duration_seconds"]=int(total_dur)
    
    #and save 
    with open(output, 'w') as fw:
        json.dump(returnarray, fw)    
    #print(frame_caption, dense_caption)

    del data, action_tensor, original_image, image,tmp,tmpa
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    return ' | '.join(tag_1),' | '.join(tag_2), frame_caption, dense_caption, gr.update(interactive = True), prediction

def set_example_video(example: list) -> dict:
    return gr.Video.update(value=example[0])

if not args.desc:
    desc = ''
else:
    desc = args.desc


inference(args.path, desc, args.opath)

try:
    inference(args.path, desc, args.opath)
except:
    print("[LOL]Out of GPU ram?")
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    image_size = 224
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([transforms.ToPILImage(),transforms.Resize((image_size, image_size)),transforms.ToTensor(),normalize])
    print("[SETUP] Retry GPU, imgsize 200, dense = 1fp15s")
    # define model
    model = tag2text_caption(pretrained="pretrained_models/tag2text_swin_14m.pth", image_size=image_size, vit='swin_b' )
    model.eval()
    model = model.to(device)
    print("[INFO] initialize caption model success!")
    # action recognition
    intern_action = load_intern_action(device)
    trans_action = transform_action()
    topil =  T.ToPILImage()
    print("[INFO] initialize InternVideo model success!")
    dense_caption_model = DenseCaptioning(device)
    dense_caption_model.initialize_model()
    print("[INFO] initialize dense caption model success!")
    print("[INFO] Loading models done")

    try:
        inference(args.path, desc, args.opath, optimize=1)
    except:
        print("[LOLZ]Out of GPU ram again? GOnna retry on CPU lol")
        device = "cpu"
        image_size = 224
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        transform = transforms.Compose([transforms.ToPILImage(),transforms.Resize((image_size, image_size)),transforms.ToTensor(),normalize])
        print("[SETUP] Obviously on CPU, imgsize 128, dense = 1fp15s")
        # define model
        model = tag2text_caption(pretrained="pretrained_models/tag2text_swin_14m.pth", image_size=image_size, vit='swin_b' )
        model.eval()
        model = model.to(device)
        print("[INFO] initialize caption model success!")

        # action recognition
        intern_action = load_intern_action(device)
        trans_action = transform_action()
        topil =  T.ToPILImage()
        print("[INFO] initialize InternVideo model success!")

        dense_caption_model = DenseCaptioning(device)
        dense_caption_model.initialize_model()
        print("[INFO] initialize dense caption model success!")
        print("[INFO] Loading models done")
        inference(args.path, desc, args.opath)
    #bot = ConversationBot()
#go back to where we start the script from
os.chdir(directory)

#with gr.Blocks(css="#chatbot {overflow:auto; height:500px;}") as demo:
#    gr.Markdown("<h1><center>Ask Anything with StableLM</center></h1>")
#    gr.Markdown(
#        """
#        Ask-Anything is a multifunctional video question answering tool that combines the functions of Action Recognition, Visual Captioning and StableLM. Our solution generates dense, descriptive captions for any object and action in a video, offering a range of language styles to suit different user preferences. It supports users to have conversations in different lengths, emotions, authenticity of language.<br>  
#        <p><a href='https://github.com/OpenGVLab/Ask-Anything'><img src='https://img.shields.io/badge/Github-Code-blue'></a></p><p>
#        """
#    )
#    
#    with gr.Row():
#        with gr.Column():
#            input_video_path = gr.inputs.Video(label="Input Video")
#            input_tag = gr.Textbox(lines=1, label="User Prompt (Optional, Enter with commas)",visible=False)
#          
#            with gr.Row():
#                with gr.Column(sclae=0.3, min_width=0):
#                    caption = gr.Button("✍ Upload")
#                    chat_video = gr.Button(" 🎥 Let's Chat! ", interactive=False)
#                with gr.Column(scale=0.7, min_width=0):
#                    loadinglabel = gr.Label(label="State")
#        with gr.Column():
#            chatbot = gr.Chatbot(elem_id="chatbot", label="gpt")
#            state = gr.State([])
#            user_tag_output = gr.State("")
#            image_caption_output = gr.State("")
#            video_caption_output  = gr.State("")
#            model_tag_output = gr.State("")
#            dense_caption_output = gr.State("")
#            with gr.Row(visible=False) as input_raws:
#                with gr.Column(scale=0.8):
#                    txt = gr.Textbox(show_label=False, placeholder="Enter text and press enter").style(container=False)
#                with gr.Column(scale=0.10, min_width=0):
#                    run = gr.Button("🏃‍♂️Run")
#                with gr.Column(scale=0.10, min_width=0):
#                    clear = gr.Button("🔄Clear️")    
#            with gr.Row():
#                example_videos = gr.Dataset(components=[input_video_path], samples=[['images/playing_guitar.mp4'], ['images/yoga.mp4'], ['images/making_cake.mp4']])
#
#    example_videos.click(fn=set_example_video, inputs=example_videos, outputs=example_videos.components)
#
#    caption.click(lambda: gr.update(interactive = False), None, chat_video)
#    caption.click(lambda: [], None, chatbot)
#    caption.click(lambda: [], None, state)    
#    caption.click(inference,[input_video_path,input_tag],[model_tag_output, user_tag_output, image_caption_output, dense_caption_output, chat_video, loadinglabel])
#
#    chat_video.click(bot.init_agent, [image_caption_output, dense_caption_output, model_tag_output], [input_raws,chatbot])
#
#    txt.submit(bot.run_text, [txt, state], [chatbot, state])
#    txt.submit(lambda: "", None, txt)
#    run.click(bot.run_text, [txt, state], [chatbot, state])
#    run.click(lambda: "", None, txt)
#
#    # clear.click(bot.memory.clear)
#    clear.click(lambda: [], None, chatbot)
#    clear.click(lambda: [], None, state)
#    
#
#
#demo.launch(server_name="0.0.0.0",enable_queue=True,)#share=True)
