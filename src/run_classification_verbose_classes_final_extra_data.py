import torch
import os
import urllib
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler, Subset, ConcatDataset
from pre.vggish_input import wavfile_to_examples
from pre.vggish import VGGish
from tqdm import tqdm
import pandas as pd
from torch import nn
from mymodel import ClassificationModel
from mydataset import AudioDataset, PseudoAudioDataset, AudioDatasetFinalClasses, AudioDataset9classes
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix
import random

import shutil
def fix(seed):
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	np.random.seed(seed)
	random.seed(seed)
fix(0)
#np.random.seed(0)
verbose2simple = {x:x for x in range(15)}
for i in range(5,11):
	verbose2simple[i] = 5
for i in range(11,15):
	verbose2simple[i] = i-5

exp_name = "simple_verbose_classes_final_extra_noiseonly2"


if not os.path.exists(os.path.join("ckpt",exp_name)):
	os.makedirs(os.path.join("ckpt",exp_name))
if not os.path.exists(os.path.join("results",exp_name)):
	os.makedirs(os.path.join("results",exp_name))

dset = AudioDatasetFinalClasses("train")

distinct_samples = len(dset)//5
idx_full = np.arange(distinct_samples)
dset2 = AudioDatasetFinalClasses("audioset",naug = 1,gen_noise = 0)
concat_training_dataset = ConcatDataset([dset, dset2])
idx_audioset = []
print(len(dset2))
for n,(_,l) in enumerate(dset2):
	if l not in [0,1,2]: # Change this to filter the data of the second training set
		idx_audioset.append(n)
	if n == len(dset2) - 1:
		break
print(idx_audioset) 
idx_audioset = [x+len(dset) for x in idx_audioset]
#idx_audioset = np.arange(len(dset2)) + len(dset)
#print(idx_audioset)
#s()
np.random.shuffle(idx_full)


_split = 0.2
len_valid = int(len(idx_full) * _split)
def get_train_val(splt):
	start = (splt-1) * len_valid
	end = (splt-1) * len_valid +  len_valid
	valid_idx_no_aug = sorted([x for x in idx_full[start:end] if x < 1600])
	valid_idx = []
	for i in range(5):
		shift = distinct_samples * i
		valid_idx += [shift + v for v in valid_idx_no_aug]
	train_idx_no_aug = np.delete(idx_full, np.arange(start, end))
	train_idx = []
	for i in range(5):
		shift = distinct_samples * i
		train_idx += [shift + v for v in train_idx_no_aug]
	train_idx += [x for x in idx_audioset]
	return train_idx, valid_idx
#print(train_idx)
#print(valid_idx)
#s()



_bs = 128
nepochs = 40


#train_loader = DataLoader(dset, batch_size = _bs,drop_last = True, shuffle = True)
#test_loader = DataLoader(dset, batch_size = 100,drop_last = False)

#criterion = nn.CrossEntropyLoss()
class LabelSmoothingCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(LabelSmoothingCrossEntropyLoss, self).__init__()
    def forward(self, x, target, smoothing=0.1):
        confidence = 1. - smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + smoothing * smooth_loss
        return loss.mean()

criterion = LabelSmoothingCrossEntropyLoss()

for fold in range(5,0,-1):
	
	#reinitialize model
	model = ClassificationModel(n_classes = 15, nheads = 4).cuda()
	optimizer = torch.optim.Adam(model.parameters(),lr = 1e-4)
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(nepochs*0.625), gamma=0.1)

	train_idx,valid_idx = get_train_val(fold)
	train_sampler = SubsetRandomSampler(train_idx)
	valid_set = Subset(dset,valid_idx)
	train_loader = DataLoader(concat_training_dataset, sampler = train_sampler , batch_size = _bs,drop_last = True)
	valid_loader = DataLoader(valid_set , batch_size = _bs,drop_last = False,shuffle=False)
	roc_baseline = 0
	best_epoch = 0
	for e in range(nepochs):
		all_loss = []
		print("Epoch:",e)

		#train
		model.train()
		for n,(batch,labels) in tqdm(enumerate(train_loader),total = len(train_loader)):
			optimizer.zero_grad()
			out = model(batch.cuda())
			loss = criterion(out,labels.cuda())
			all_loss.append(loss)

			loss.backward()
			optimizer.step()
			
		scheduler.step()
		print("loss:",sum(all_loss)/len(all_loss))
		if e % 10 == 9:
			torch.save(model.state_dict(),f"./ckpt/{exp_name}/Epoch{e}.ckpt")
		#eval 
		y_true = torch.zeros(len(valid_idx),)
		y_score = torch.zeros(len(valid_idx),10)
		model.eval()
		with torch.no_grad():
			corrects = 0
			for n,(batch,labels) in tqdm(enumerate(valid_loader),total = len(valid_loader)):
				
				out = model(batch.cuda())
				loss = criterion(out,labels.cuda())
				corrects += (torch.argmax(out,dim = -1).squeeze() == labels.cuda()).sum()
				y_true[ n * _bs : n * _bs+_bs] = torch.FloatTensor([verbose2simple[x.item()] for x in labels]) #F.one_hot(labels, num_classes=16)
				y_score[ n * _bs : n * _bs+_bs,5] = torch.sum(out[:,5:11],dim = -1)
				y_score[ n * _bs : n * _bs+_bs,:5] = out[:,:5]
				y_score[ n * _bs : n * _bs+_bs,6:] = out[:,11:]
		y_true = y_true[:len(valid_idx)//5]
		y_score = y_score.reshape(5,-1,10)
		y_score = y_score.mean(dim = 0)
		#print(set(y_true.tolist()))		
		#print(y_true,y_score)	
		roc = roc_auc_score(y_true, y_score, multi_class="ovr", average="weighted")
		if roc > roc_baseline:
			best_epoch = e
			roc_baseline = roc
			torch.save(model.state_dict(),f"./ckpt/{exp_name}/Best{fold}.ckpt")
		print("precision",corrects/len(valid_idx))
		print("roc_auc_score",roc)
	print("best roc_auc_score",roc_baseline)
	print("best epoch",best_epoch)
	print(valid_idx)
	for temperature in [1,2,5,10,20,50,100]: 
		model = ClassificationModel(temp = temperature,n_classes = 15, nheads = 4).cuda()
		model.load_state_dict(torch.load(f"./ckpt/{exp_name}/Best{fold}.ckpt"))
		model.eval()
		y_true = torch.zeros(len(valid_idx),)
		y_score = torch.zeros(len(valid_idx),10)
		with torch.no_grad():
			corrects = 0
			for n,(batch,labels) in tqdm(enumerate(valid_loader),total = len(valid_loader)):
				
				out = model(batch.cuda())
				loss = criterion(out,labels.cuda())
				corrects += (torch.argmax(out,dim = -1).squeeze() == labels.cuda()).sum()
				y_true[ n * _bs : n * _bs+_bs] = torch.FloatTensor([verbose2simple[x.item()] for x in labels]) #F.one_hot(labels, num_classes=16)
				y_score[ n * _bs : n * _bs+_bs,5] = torch.sum(out[:,5:11],dim = -1)
				y_score[ n * _bs : n * _bs+_bs,:5] = out[:,:5]
				y_score[ n * _bs : n * _bs+_bs,6:] = out[:,11:]
		y_true = y_true[:len(valid_idx)//5]
		y_score = y_score.reshape(5,-1,10)
		y_score = y_score.mean(dim = 0)
		_,y_preds = torch.max(y_score,dim = -1)
		#print(y_preds)
		best_roc = roc_auc_score(y_true, y_score, multi_class="ovr", average="weighted")
		print("best roc_auc_score",best_roc)
		print(confusion_matrix(y_true, y_preds))
		if temperature == 1:
			temp1_roc = best_roc
			if not os.path.exists(f"./ckpt/{exp_name}/FOLD{fold}_{best_roc}"):
				os.makedirs(f"./ckpt/{exp_name}/FOLD{fold}_{best_roc}")
			try:
				shutil.copy(f"./ckpt/{exp_name}/Best.ckpt", f"./ckpt/{exp_name}/FOLD{fold}_{best_roc}")
			except:
				pass
			try:
				shutil.copytree(f"./src", f"./ckpt/{exp_name}/FOLD{fold}_{best_roc}/src")
			except:
				pass
			with open(f"./ckpt/{exp_name}/FOLD{fold}_{best_roc}/best_temperature.csv","w") as f:
				f.write("temperature,best_roc\n")
				f.write(f"{temperature},{best_roc}\n")
		else:
			with open(f"./ckpt/{exp_name}/FOLD{fold}_{temp1_roc}/best_temperature.csv","a") as f:
				f.write(f"{temperature},{best_roc}\n")