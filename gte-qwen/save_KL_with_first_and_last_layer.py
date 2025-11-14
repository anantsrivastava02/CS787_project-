import os
os.environ['TORCH_COMPILE_DISABLE'] = '1'

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import argparse
import torch.nn.functional as F
import os
import json
from tqdm import tqdm
import pickle

# Fix for DynamicCache compatibility issue
try:
    from transformers.cache_utils import DynamicCache
    if not hasattr(DynamicCache, 'get_usable_length'):
        DynamicCache.get_usable_length = lambda self, seq_length: self.get_seq_length()
except ImportError:
    pass

def last_token_pool(last_hidden_states,
                 attention_mask):
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device='cpu'), sequence_lengths]
    
max_length = 512


pretrained_model_name_or_path = 'Alibaba-NLP/gte-Qwen2-1.5B-instruct'
which_embedding='gte_qwen2_1.5B_KL_with_first_and_last_layer'

tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path, trust_remote_code=True).to(device)

save_dir = f'save/{which_embedding}/'
def get_kl(model,input_texts):
    batch_dict = tokenizer(input_texts, max_length=max_length, padding=True, truncation=True, return_tensors='pt')
    batch_dict = {k: v.to(device) for k, v in batch_dict.items()}
    with torch.no_grad():
        outputs = model(**batch_dict, output_hidden_states=True, use_cache=False)
        last_logits = model.lm_head(outputs.hidden_states[-1]).squeeze()
        first_logits = model.lm_head(outputs.hidden_states[0]).squeeze()
    kls = []
    for i in range(1,len(outputs.hidden_states)-1):
        with torch.no_grad():
            middle_logits = model.lm_head(outputs.hidden_states[i]).squeeze()
        kls.append(F.kl_div(F.log_softmax(middle_logits, dim=-1), F.softmax(first_logits, dim=-1), reduction='batchmean').item()+
                   F.kl_div(F.log_softmax(middle_logits, dim=-1), F.softmax(last_logits, dim=-1), reduction='batchmean').item())
    return kls

data_dir = 'dataset/processed_data/'
test_datasets = {}
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
for file_name in os.listdir(data_dir):
    if not os.path.exists(save_dir+file_name.split('.')[0]+'.pkl'):
        print(file_name)
        result_data = []
        test_datasets[file_name] = {'data':[],'label':[]}
        with open(data_dir+file_name, 'r') as f:
            data = json.load(f)
        kls = []
        for text_info in tqdm(data):
            text = text_info['text']
            result = text_info['result']
            prompt = text
            kl = get_kl(model,[text])
            kls.append(kl)
            if len(kls)>=300:
                break
        print(save_dir+file_name.split('.')[0]+'.pkl')
        pickle.dump(kls, open(save_dir+file_name.split('.')[0]+'.pkl', 'wb'))
