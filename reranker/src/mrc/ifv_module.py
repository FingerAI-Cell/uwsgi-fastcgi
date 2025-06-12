import os
import time
import torch
import json
import gdown

import pytorch_lightning as pl

from munch import munchify
from torch import nn
import torch.nn.functional as F

from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForMaskedLM)


class IFVModule(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        if self.args.plm == 'kobigbird':
            self.tokenizer = AutoTokenizer.from_pretrained('monologg/kobigbird-bert-base')
            self.plm = AutoModel.from_pretrained('monologg/kobigbird-bert-base')
            if self.args.init_scale != 'base':
                self.plm = self.scale_plm(size_to=self.args.init_scale)
        elif self.args.plm == 'koroberta':
            if self.args.init_scale == 'base':
                self.tokenizer = AutoTokenizer.from_pretrained('klue/roberta-base')
                self.plm = AutoModelForMaskedLM.from_pretrained('klue/roberta-base').roberta
            elif self.args.init_scale == 'large':
                self.tokenizer = AutoTokenizer.from_pretrained('klue/roberta-large')
                self.plm = AutoModelForMaskedLM.from_pretrained('klue/roberta-large').roberta
        
        self.pooling_method = self.args.pooling_method
        self.mrc_linear = nn.Linear(self.plm.encoder.layer[-1].output.dense.weight.size(0), 2)
        self.ans_linear = nn.Linear(self.plm.encoder.layer[-1].output.dense.weight.size(0), 2)
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)

        self.after_sanity_check = False
        self.best_scores = {val_darg: {metric: 0. for metric in ['ans_acc', 'ans_Mf1', 'mrc_em', 'mrc_f1']}
            for val_darg in self.args.val_datasets}

    def forward(self, input_ids, token_type_ids, attention_mask):
        if self.args.plm in ['kobigbird', 'koelectra', 'electra']:
            output = self.plm(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        elif self.args.plm == 'koroberta':
            output = self.plm(input_ids) # token_type_embeddings is not (2, 1024) but (1, 1024)
        hidden = output['last_hidden_state'] # (bat, len, dim)

        mrc_logits = self.mrc_linear(hidden) # (bat, len, 2)
        start_logits, end_logits = mrc_logits.split(1, dim=-1) # (bat, len, 1) each
        start_logits = start_logits.squeeze(-1).contiguous() # (bat, len)
        end_logits = end_logits.squeeze(-1).contiguous() # (bat, len)

        if self.pooling_method == 'cls':
            pooled = hidden[:,0,:] # (bat, dim)
        elif self.pooling_method == 'avg':
            pooled = torch.mean(input=hidden, dim=1) # (bat, dim)
        elif self.pooling_method == 'max':
            pooled, _ = torch.max(input=hidden, dim=1) # (bat, dim)
        ans_logits = self.ans_linear(pooled) # (bat, 2)

        return {
            'start_logits': start_logits,
            'end_logits': end_logits,
            'ans_logits': ans_logits}


def download_checkpoints(output_path, gdrive_id=''):
    # init
    sleep_count = 3
    sleep_sec = 60
    opath = os.path.abspath(output_path)

    # download process
    n = 0
    if not os.path.exists(opath):
        # 만약 load하려는 데이터가 존재하지 않는다면
        fname = os.path.basename(opath)
        dpath = os.path.dirname(opath)
        if os.path.exists(dpath):    # os.listdir에서 에러를 방지하기 위한 조건
            if fname in [fn[:len(fname)] for fn in os.listdir(dpath)]:
                # 만약 다른 프로세스에 의해 다운로드 중일 경우
                # epoch=39-step=331346.ckptn09i9gs9tmp 이런식으로 임시 파일 형태의 이름으로 존재함
                while True:
                    if n > sleep_count:
                        print(f"[re-download] - {fname}")
                        break
                    time.sleep(sleep_sec)
                    if os.path.exists(opath):
                        print(f"[ready] to load - {fname}")
                        break
                    print(f"[sleep] wait other process ({n}/{sleep_count})")
                    n += 1
        
        if (n == 0) or (n==3):
            # <가능한 Case>
            # 1. n == 0 : load하려는 데이터가 존재하지 않고 && 다른 프로세스에 의해 다운로드 중도 아니거나
            # 2. n == 3 : load하려는 데이터가 존재하지 않지만 && 다른 프로세스에 의해 다운로드 중인데, 재다운이 필요하다고 판단되는 경우
            url = f"https://drive.google.com/uc?id={gdrive_id}"
            output = output_path
            os.makedirs(dpath, exist_ok=True)
            gdown.download(url, output, quiet=False)

def get_model_config(config_path, plm='koroberta', infer_batch_size=10):
    config_dict = json.load(open(config_path, 'r'))
    config_args = munchify(config_dict)
    config_args.plm = plm
    config_args.infer_batch_size = infer_batch_size
    config_args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return config_args

def get_model(config_args, model_path):
    # load model
    model = IFVModule(config_args)
    state_dict = torch.load(model_path)['state_dict']
    model.load_state_dict(state_dict)
    model.to(config_args.device)
    model.eval()
    model.to(dtype=torch.float16)  # fp16 precision

    return model 