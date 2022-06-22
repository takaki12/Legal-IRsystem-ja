# Sentence-BERTファインチューニング用

import pandas as pd
import random
import glob
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from transformers import BertJapaneseTokenizer, BertModel
import pytorch_lightning as pl
from pytorch_lightning.trainer.supporters import CombinedLoader

import numpy as np

# GPUの設定
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# 日本語の事前学習モデル
MODEL_NAME = 'cl-tohoku/bert-base-japanese-whole-word-masking'

# トークナイザ―
tokenizer = BertJapaneseTokenizer.from_pretrained(MODEL_NAME)

# データの読み込み
file = open('data_dir','r')
# data frame (pandasじゃなくてもいい)
# a:アンカーテキスト, p:ポジティブテキスト(≒a), n:ネガティブテキスト(!≒a)
data_val = pd.read_csv(file, sep='\t', encoding='utf-8', names=['a','p','n'])

# validationデータの作成
max_length = 16
data_transforms_val = []
texts_a = data_val['a'].values
texts_p = data_val['p'].values
texts_n = data_val['n'].values
dataset_for_loader=[]

for text_a, text_p, text_n in zip(texts_a, texts_p, texts_n):
    encoding = {}
    for text, f in zip([text_a, text_p, text_n], ['a', 'p', 'n']):
        encoding_sub = tokenizer(
            text,
            max_length=max_length,
            padding='max_length',
            truncation=True
        )
        
        encodin_sub = { k+'_'+f: torch.tensor(v).cuda() for k, v in encoding_sub.items() }
        encoding.update(encodin_sub)
    dataset_for_loader.append(encoding)

# データセット分割する (とりあえず、train,val,test=8:1:1)
random.shuffle(dataset_for_loader)
n = len(dataset_for_loader)
n_train = int(0.6*n)
n_val = int(0.2*n)
dataset_train = dataset_for_loader[:n_train]
dataset_val = dataset_for_loader[n_train:n_train + n_val]
dataset_test = dataset_for_loader[n_train+n_val:]

dataloader_train = DataLoader(
    dataset_train, batch_size=32, shuffle=False
)
dataloader_val = DataLoader(
    dataset_val, batch_size=32, shuffle=False
)
dataloader_test = DataLoader(
    dataset_test, batch_size=32, shuffle=False
)


################################################################################
### SentenceBERTモデルの実装

# 平均でプーリングする。
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

class SentenceBERT(pl.LightningModule):

    def __init__(self, model_name, lr):
        # model_name: Transformersのモデルの名前
        # lr: 学習率

        super().__init__()

        # 例えば、self.hparams.lrでlrにアクセスできる。
        # チェックポイント作成時にも自動で保存される。
        self.save_hyperparameters() 

        # BERTのロード
        # p/n用それぞれで用意するから2つつくる。
        self.bert_sc1 = BertModel.from_pretrained(
            model_name
        )
        self.bert_sc1.cuda()

        self.bert_sc2 = BertModel.from_pretrained(
            model_name
        )
        self.bert_sc2.cuda()

        self.triplet_loss = torch.nn.TripletMarginWithDistanceLoss(distance_function=torch.nn.PairwiseDistance(p=2), margin=1.0)
        #self.triplet_loss = torch.nn.TripletMarginWithDistanceLoss(distance_function=torch.nn.CosineSimilarity(), margin=1.0)
        self.cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

    # 学習データのミニバッチ(`batch`)が与えられた時に損失を出力する関数を書く。
    # batch_idxはミニバッチの番号であるが今回は使わない。
    def training_step(self, batch, batch_idx):
        # データセットのa/p/nそれぞれで埋め込み計算する。
        output1 = mean_pooling(self.bert_sc1(attention_mask=batch['attention_mask_a'], 
                                             input_ids=batch['input_ids_a'], 
                                             token_type_ids=batch['token_type_ids_a']), 
                               batch['attention_mask_a'])
        output2 = mean_pooling(self.bert_sc2(attention_mask=batch['attention_mask_p'], 
                                             input_ids=batch['input_ids_p'], 
                                             token_type_ids=batch['token_type_ids_p']), 
                               batch['attention_mask_p'])
        output3 = mean_pooling(self.bert_sc2(attention_mask=batch['attention_mask_n'], 
                                             input_ids=batch['input_ids_n'], 
                                             token_type_ids=batch['token_type_ids_n']), 
                               batch['attention_mask_n'])
        loss = self.triplet_loss(output1,output2,output3)
        self.log('train_loss', loss) # 損失を'train_loss'の名前でログをとる。
        return loss

    # 検証データのミニバッチが与えられた時に、
    # 検証データを評価する指標を計算する関数を書く。
    def validation_step(self, batch, batch_idx):
        output1 = mean_pooling(self.bert_sc1(attention_mask=batch['attention_mask_a'], 
                                             input_ids=batch['input_ids_a'], 
                                             token_type_ids=batch['token_type_ids_a']), 
                               batch['attention_mask_a'])
        output2 = mean_pooling(self.bert_sc2(attention_mask=batch['attention_mask_p'], 
                                             input_ids=batch['input_ids_p'], 
                                             token_type_ids=batch['token_type_ids_p']), 
                               batch['attention_mask_p'])
        output3 = mean_pooling(self.bert_sc2(attention_mask=batch['attention_mask_n'], 
                                             input_ids=batch['input_ids_n'], 
                                             token_type_ids=batch['token_type_ids_n']), 
                               batch['attention_mask_n'])
        val_loss = self.triplet_loss(output1,output2,output3)
        self.log('val_loss', val_loss) # 損失を'val_loss'の名前でログをとる。

    # 学習に用いるオプティマイザを返す関数を書く。
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


################################################################################


# 学習時にモデルの重みを保存する条件を指定
checkpoint = pl.callbacks.ModelCheckpoint(
    monitor='val_loss',
    mode='min',
    save_top_k=1,
    save_weights_only=True,
    dirpath='./model/',
)

# 学習の方法を指定
trainer = pl.Trainer(
    gpus=1, # 個数,[num]で番号。
    max_epochs=3, # もっと多くてよさそう
    auto_lr_find=True,
    callbacks = [checkpoint]
)

# モデルをロード
model = SentenceBERT(model_name=MODEL_NAME, lr=1e-5)

# データからlearning_rateを求めてくれるらしい。
lr_finder = trainer.tuner.lr_find(model, dataloader_train, dataloader_val)
model.hparams.lr = lr_finder.suggestion()

# テストデータの評価用
def evaluation_testdata(dataloader_test,model):
    losses = []
    sim_diffs = []
    prc = []
    for batch in dataloader_test:
        output1 = mean_pooling(model.bert_sc1(attention_mask=batch['attention_mask_a'], 
                                input_ids=batch['input_ids_a'], 
                                token_type_ids=batch['token_type_ids_a']), 
                                batch['attention_mask_a'])
        output2 = mean_pooling(model.bert_sc2(attention_mask=batch['attention_mask_p'], 
                                input_ids=batch['input_ids_p'], 
                                token_type_ids=batch['token_type_ids_p']), 
                                batch['attention_mask_p'])
        output3 = mean_pooling(model.bert_sc2(attention_mask=batch['attention_mask_n'], 
                                input_ids=batch['input_ids_n'], 
                                token_type_ids=batch['token_type_ids_n']), 
                                batch['attention_mask_n'])
        loss = model.triplet_loss(output1,output2,output3)
        losses.extend([float(loss.cpu())])
        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        sim_diff = cos(output1, output2)-cos(output1, output3)
        sim_diffs.extend(np.array(sim_diff.cpu()))
        prc.extend([float(torch.sum(cos(output1, output2)>cos(output1, output3)).cpu())])

    print("loss:{0}".format(np.array(losses).mean()))
    print("cos_diff:{0}".format(np.array(sim_diffs).mean()))
    print("presicion:{0}".format(np.array(prc).sum()/len(dataloader_test)))

# ファインチューニング前に試してみる
evaluation_testdata(dataloader_test,model)

# ファインチューニングの実行!
trainer.fit(model, dataloader_train, dataloader_val)

# ファインチューニング後のモデルのロード
best_model_path = checkpoint.best_model_path
print('ベストモデルのファイル: ', checkpoint.best_model_path)
print('ベストモデルの検証データに対する損失: ', checkpoint.best_model_score)

# テストデータによる評価
test = trainer.test(dataloaders=dataloader_test)

# モデルのロード
model = SentenceBERT.load_from_checkpoint(
    best_model_path
)

# 保存
model.bert_sc.save_pretrained('./model_transformers')