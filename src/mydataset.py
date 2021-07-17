from torch.utils.data import Dataset
import os
import torch
import pandas as pd
from tqdm import tqdm
import soundfile as sf
import numpy as np
import colorednoise as noise
import os
from pre.vggish_input import waveform_to_examples


_data_dir_dictionary = {}
_data_dir_dictionary["train"] = "./data/Final_Training_Dataset"
_data_dir_dictionary["AudioSet"] = "./data/AudioSet"



class AudioDatasetFinalClasses(Dataset):
	
	def __init__(self,fold = "test",naug = 5, pseudo_label = "",gen_noise = 250,balance = 0):
		self.fold = fold
		switch_set_dir = _data_dir_dictionary
		switch_feats_dir = os.path.join(_data_dir_dictionary,"feats")
		switch_data_dir = os.path.join(switch_set_dir,"raw")
		
		files = sorted(os.listdir(switch_data_dir))
		self.hids_aug = {i:torch.zeros((len(files),5,1,96,64)) for i in range(1,naug+1)}
		for i in range(1,naug+1):
			#find cache
			if not os.path.isfile(os.path.join(switch_feats_dir,f"all_melfeatures{i}.tensor")):
				print("no found features")
				#self.hids = torch.zeros((len(files),5,1,96,64))
				for n,f in tqdm(enumerate(files),total = len(files)):
					if f.startswith("_"):
						continue
					wav_data, sr = sf.read(os.path.join(switch_data_dir,f), dtype='int16',start = (8000//naug)*(i-1))
					assert wav_data.dtype == np.int16, 'Bad sample type: %r' % wav_data.dtype
					samples = wav_data / 32768.0  # Convert to [-1.0, +1.0]
					#print(wav_data.shape)
					try:
						mel = waveform_to_examples(samples, sr, return_tensor=True)
					except:
						print(f"Error converting sample No. {n}")
						mel =  torch.zeros((5,1,96,64))
					#print(mel.shape)
					#print(mel[1][0][44][43])
					

					self.hids_aug[i][n,:mel.shape[0],:,:,:] = mel

				torch.save(self.hids_aug[i],os.path.join(switch_feats_dir,f"all_melfeatures{i}.tensor"))
			# use cache
			else:
				self.hids_aug[i] = torch.load(os.path.join(switch_feats_dir,f"all_melfeatures{i}.tensor")).detach()
				print("load from save")#,self.hids)
		
		if fold == "train" or fold == "AudioSet":
			#Get labels and generate noise as class "Other"
			metafile = os.path.join(switch_set_dir,"meta_train.csv")
			df = pd.read_csv(metafile)
			df = df.sort_values(by=['Filename'])
			df["verbose_label"] = df["Label"].astype(str) + df["Remark"]
			ll = list(df["verbose_label"])
			ll.append("5Other")
			ll.append("8Music_Instrument")
			typ = {e:i for i,e in enumerate(sorted(set(ll)))}
			df["verbose_label"] = df["verbose_label"].apply(lambda x : typ[x])
			self.labels = torch.Tensor(list(df["verbose_label"])).type(torch.LongTensor)
			print("labels",set(list(df["verbose_label"])))
			maxlabel = max(set(list(df["verbose_label"])))

			# Augment noise as class "Others"
			if gen_noise != 0:
				beta = 1 # the exponent
				samples = 40000 # number of samples to generate
				sr = 8000
				gen_noise_per_fold = gen_noise // naug
				for i in range(1,1+naug):
					for j in range(gen_noise_per_fold):
						
						y = noise.powerlaw_psd_gaussian(beta, samples)
						y /= 1.5
						mel = waveform_to_examples(y, sr, return_tensor=True)
						mel = mel.unsqueeze(0).detach()
						self.hids_aug[i] = torch.cat([self.hids_aug[i],mel],dim = 0)
				noise_labels = torch.Tensor([typ["5Other"] for i in range(gen_noise_per_fold)]).type(torch.LongTensor)
				self.labels = torch.cat([self.labels,noise_labels],dim =0)

			for i in range(1,naug+1):
				print(self.hids_aug[i].shape)

			print(self.labels.shape)

	def __len__(self):
		if self.fold == "train" or "test":
			return len(self.hids_aug[1]) * len(self.hids_aug)
		else:
			return len(self.hids_aug[1]) #* len(self.hids_aug)

	def __getitem__(self,idx):
		aug = idx // len(self.hids_aug[1])+1
		iidx = idx % len(self.hids_aug[1])
		return self.hids_aug[aug][iidx], self.labels[iidx]

