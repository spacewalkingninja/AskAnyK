import os
directory = os.getcwd()
#no meu codigo emos que enetrar na pasta onde  estamos a exefceutr o codigo
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)
#basicamente temos que OS.CHDIR(PAST_DO_FICHEIRO.PY)
#os.chdir == cd /pasta no cmd escondido do python
import json
import numpy as np
import random
import torch
import torchvision.transforms as transforms
from PIL import Image

#antes disto temos q tar na pasta do ficheiro.py qaue corre o programa
from models.tag2text import tag2text_caption
from util import *
import gradio as gr
#from stablelm import *
#from chatbot import *
import argparse
from load_internvideo import *
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #mas n faz sentido cpu pq isso nnc sai do sitio
from models.grit_model import DenseCaptioning


parser = argparse.ArgumentParser(description='vid 2 text')
parser.add_argument('--desc', dest='desc', type=str, default=None, help="Description")
parser.add_argument('--path', dest='path', type=str, default='./img', help="INput path of image(s) folder")
parser.add_argument('--json', dest='opath', type=str, default='./img_caption.json', help="Output path with filename.json")

args = parser.parse_args()



image_size = 384
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
transform = transforms.Compose([transforms.ToPILImage(),transforms.Resize((image_size, image_size)),transforms.ToTensor(),normalize])

if device != 'cpu':
    print("[SETUP] Running img understanding on GPU")    
    torch.cuda.set_per_process_memory_fraction(1.0, 0) #n ]e estritamente necessario pq ele automaticamente aloca o maximo usavel mas why not ne 
    torch.cuda.empty_cache()
    total_memory = torch.cuda.get_device_properties(0).total_memory
    print(f"[SETUP]{total_memory} allocated to gpu")
else:
    print("[SETUP] Running img understanding on CPU")

# action recognition
intern_action = load_intern_action(device) #mas podemos por no GPU
trans_action = transform_action()
topil =  T.ToPILImage()
print("[INFO] initialize InternAction FOR IMG model success!")


# define model
model = tag2text_caption(pretrained="pretrained_models/tag2text_swin_14m.pth", image_size=image_size, vit='swin_b' )
model.eval()
model = model.to(device)
print("[INFO] initialize caption model success!")

print("[INFO] Loading models done")


#bot = ConversationBot()

def inference(img_folder_path, input_tag='', output='./img_caption.json', progress=gr.Progress()):
    print("Watching images...")
    data = load_images_decord(img_folder_path)
    progress(0.2, desc="Loading Images")

    #temp vars
    tmp, tmpa = [], []
    #action index is the array of all frames in numpy format
    action_index = np.linspace(0, len(data)-1, 8).astype(int)
    #for all frames in data
    for i, img in enumerate(data):
        if i in action_index:
            tmpa.append(Image.fromarray((img.transpose(1, 2, 0) * 255).astype(np.uint8)))

    #este gajo faz uma descricao bue vagas tipo @a kite flying@ ou @a womand dancing@
    action_tensor = trans_action(tmpa)
    TC, H, W = action_tensor.shape
    action_tensor = action_tensor.reshape(1, TC//3, 3, H, W).permute(0, 2, 1, 3, 4).to(device)
    with torch.no_grad():
        prediction = intern_action(action_tensor)
        prediction = F.softmax(prediction, dim=1).flatten()
        prediction = kinetics_classnames[str(int(prediction.argmax()))]
        print(prediction) #frase bue vaga acerca do video

    #aqui podemos tentar
    del action_tensor #libertar ram deste model

    print("Step 1/4")
    # InternVideo
    #print(f'data:{len(data)}')
    #action_index = np.linspace(0, len(data)-1, 8).astype(int)
    #tmp,tmpa = [],[]
    #for i,img in enumerate(data):
    #    tmp.append(transform(img).to(device).unsqueeze(0))
    #    if i in action_index:
    #        tmpa.append(topil(img))
    tf=1
    print(f'data:{len(data)} at {tf} seconds per frame viewed by the model')
    tmp, tmpa = [], []
    frame_counter = 0
    batch_frames = []
    batch_counter = 0
    caption, tag_predict = [], []
    captions,tag_predicts = [], []
    #input_tags ajuda ao gajo entender o q ta a ver, mas nos n usamos nd disso, ele q se safe sozinho
    if input_tag == '' or input_tag == 'none' or input_tag == 'None':
        input_tag_list = None
    else:
        input_tag_list = []
        input_tag_list.append(input_tag.replace(',',' | '))

    #aqui vamos ver o video frame por frame no TF que definimos   | TF= X segundos por cada frame visualizado
    for i, img in enumerate(data):
        if i % tf == 0:
            tmp.append(transform(img).to(device).unsqueeze(0))
            batch_frames.append(img)
            frame_counter += 1
            if frame_counter % 42 == 0 or frame_counter == len(data):
                print(f"[INFO]Batch {batch_counter}, ends at {frame_counter} imgs")
                # process batch
                image = torch.cat(tmp).to(device)
                model.threshold = 0.68

                with torch.no_grad():
                    caption, tag_predict = model.generate(image,tag_input = input_tag_list,max_length = 50, return_tag_predict = True)
                    captions += caption
                    tag_predicts += tag_predict


                    # do something with the caption
                    # ...
                # reset lists and counters
                tmp = []
                tmpa = []
                batch_frames = []
                batch_counter += 1
    print("Step 2/4")
    # Video Caption
    #image = torch.cat(tmp).to(device)   
    #model.threshold = 0.68
    #if input_tag == '' or input_tag == 'none' or input_tag == 'None':
    #    input_tag_list = None
    #else:
    #    input_tag_list = []
    #    input_tag_list.append(input_tag.replace(',',' | '))
    #with torch.no_grad():
    #    caption, tag_predict = model.generate(image,tag_input = input_tag_list,max_length = 50, return_tag_predict = True)
    #    progress(0.6, desc="Watching Videos")
        #frame_caption = ' '.join([f"Second {i+1}:{j}."+str(dcs.get(str(i+1), ""))+"\n" for i,j in enumerate(caption)])
    #if input_tag_list == None:
    #    tag_1 = set(tag_predict)
    #    tag_2 = ['none']
    #else:
    #    _, tag_1 = model.generate(image,tag_input = None, max_length = 50, return_tag_predict = True)
    #    tag_2 = set(tag_predict)

    del tmp, tmpa, image
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    print("Step 3/4")  
    # dense caption
    dense_caption_model = DenseCaptioning(device)
    dense_caption_model.initialize_model()
    print("[INFO] initialize dense caption model success!")

    dense_caption = []
    frame_per_second = 5
    dense_index = np.arange(0, len(data)-1, frame_per_second)
    original_images = data[dense_index,:,:,::-1]
    dcs = {}
    dcc = 0
    dcl = len(original_images)
    with torch.no_grad():
        for original_image in original_images:
            dcc +=1
            print(f'{dcc}/{dcl}')
            dense_caption.append(dense_caption_model.run_caption_tensor(original_image))
        #dense_caption = ' '.join([f"Second {i+1} : {j}.\n" for i,j in zip(dense_index,dense_caption)])
        for i,j in zip(dense_index,dense_caption):
            key = f"{i+1}"
            value = f"\n Extra visual information: {j}.\n"
            dcs[key] = value

    frame_caption = ""
    prev_caption = ""
    start_time = 0
    end_time = 0
    last_valid_dcs = ''
    for i, j in enumerate(captions):
        current_caption = f"{j}."
        current_dcs = dcs.get(f"{i+1}", "")
        if len(current_dcs) > 0:
            last_valid_dcs = current_dcs
        if current_caption == prev_caption:
            end_time = i+1
        else:
            if prev_caption:
                frame_caption += f"Image {i}: {last_valid_dcs}\n"
            start_time = i+1
            end_time = i+1
            prev_caption = current_caption
    if prev_caption:
        frame_caption += f"Image {i}: {current_dcs}\n"
    total_dur = end_time
    frame_caption += f"| Total: {total_dur} image(s).\n"

    progress(0.8, desc="Understanding IMGs")

    tag_predicts = set(tag_predicts)
        
    print("[INFO]" + img_folder_path + " Analyzed")
    print("[IMG_TRANSCRIPT]")
    print(frame_caption)
    print("[TAGS] "+ str( ' | '.join(tag_predicts) ))


    returnarray = {}
    returnarray["dense_caption"]=dcs
    returnarray["frame_caption"]=' '.join([f"Image {i+1}:{j}.\n" for i,j in enumerate(caption)])
    returnarray["prompt"]=frame_caption
    returnarray["tags"]=str( ' | '.join(tag_predicts))
    returnarray["total_images"]=int(total_dur)
    returnarray["prediction"]=prediction
    
    #and save 
    with open(output, 'w') as fw:
        json.dump(returnarray, fw)    
    #print(frame_caption, dense_caption)

    del data, original_image
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    return returnarray['tags'], frame_caption, dense_caption, gr.update(interactive = True), prediction

def set_example_video(example: list) -> dict:
    return gr.Video.update(value=example[0])

if not args.desc:
    desc = ''
else:
    desc = args.desc

inference(args.path, desc, args.opath)

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
#            input_img_folder_path = gr.inputs.Video(label="Input Video")
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
#                example_videos = gr.Dataset(components=[input_img_folder_path], samples=[['images/playing_guitar.mp4'], ['images/yoga.mp4'], ['images/making_cake.mp4']])
#
#    example_videos.click(fn=set_example_video, inputs=example_videos, outputs=example_videos.components)
#
#    caption.click(lambda: gr.update(interactive = False), None, chat_video)
#    caption.click(lambda: [], None, chatbot)
#    caption.click(lambda: [], None, state)    
#    caption.click(inference,[input_img_folder_path,input_tag],[model_tag_output, user_tag_output, image_caption_output, dense_caption_output, chat_video, loadinglabel])
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
