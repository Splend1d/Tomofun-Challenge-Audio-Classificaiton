#!/usr/bin/env python

import os
import time
from tqdm import tqdm
import numpy as np
from argparse import ArgumentParser
from pprint import pformat
import pandas as pd
import logging

import torch
#import librosa 
from torch.utils.data import DataLoader

#from dataset import SoundDataset_test
import soundfile as sf
from config import ParameterSetting
#from models import VGGish
from metrics import cfm, classification_report, roc_auc
#import flask
import json 

from mymodel import ClassificationModel
from vggish_input import waveform_to_examples

logger = logging.getLogger(__file__)



def wav_read(wav_file,percentile=1,naug=5):
    wav_data, sr = sf.read(wav_file, dtype='int16',start = (8000//naug)*(percentile))
    return wav_data, sr

def inference_single_audio(path,percentile=1): 
    wav_data, sr = wav_read(path,percentile)
    return wav_data, sr


def preprocessing(params, wav_data, sr):
    """Convert wav_data to log mel spectrogram.
        1. normalize the wav_data
        2. convert the wav_data into mono-channel
        3. resample the wav_data to the sampling rate we want
        4. compute the log mel spetrogram with librosa function
    Args:
        wav_data: An np.array indicating wav data in np.int16 datatype
        sr: An integer specifying the sampling rate of this wav data
    Return:
        inpt: An np.array indicating the log mel spectrogram of data
    """
        # normalize wav_data
    # if params.normalize == 'peak':
    #     samples = wav_data/np.max(wav_data)
    # elif params.normalize == 'rms':
    #     rms_level = 0
    #     r = 10**(rms_level / 10.0)
    #     a = np.sqrt((len(wav_data) * r**2) / np.sum(wav_data**2))
    #     samples = wav_data * a
    # else:
    samples = wav_data / 32768.0

        # convert samples to mono-channel file
    if len(samples.shape) > 1:
        samples = np.mean(samples, axis=1)

        # resample samples to 8k
    if sr != params.sr:
        samples = resampy.resample(samples, sr, params.sr)

    # transform samples to mel spectrogram
    # inpt_x = 500
    # spec = librosa.feature.melspectrogram(samples, sr=params.sr, n_fft=params.nfft, hop_length=params.hop, n_mels=params.mel)
    # spec_db = librosa.power_to_db(spec).T
    # spec_db = np.concatenate((spec_db, np.zeros((inpt_x - spec_db.shape[0], params.mel))), axis=0) if spec_db.shape[0] < inpt_x else spec_db[:inpt_x]
    # inpt = np.reshape(spec_db, (1, spec_db.shape[0], spec_db.shape[1]))
    mel = waveform_to_examples(samples, sr, return_tensor=True)
    if mel.shape[0] != 5:
        mel_pad = torch.zeros(5,*mel.shape[1:])
        mel_pad[:min(5,mel.shape[0])] = mel
        return mel_pad

    return mel

exp_name = "simple_verbose_classes_final"

def load_model(fold): 
    prefix = f'./ckpt/{exp_name}'
    model_name = f"Best{fold}.ckpt"
    model_path = os.path.join(prefix, model_name)   
    #model = None
    model = ClassificationModel(temp = 5, n_classes = 15, nheads = 4)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    model = model.to(device)
    return model 


params = ParameterSetting(sr=8000,nfft=200, hop=80, mel=64, normalize=None, preload=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
models = ["NULL"]
for fold in range(1,6):
    models.append(load_model(fold))

path = './data/AudioSet/raw/_1xbC-dOrAs_1.wav'

for i in range(5):
    wav,sr = inference_single_audio(path,percentile = i)
    spec = preprocessing(params, wav, sr)
    spec = spec.to(device)
    if i == 0:
        specs = torch.zeros(5,*spec.shape).to(device)
    #print(specs.shape, spec.shape)
    specs[i][:] = spec[:]

ret_tensor = torch.zeros(1,10).to(device)
for fold in range(1,6):
    
    #wav = torch.tensor([wav])
    #spec = wav.to(device)
    outputs = models[fold](specs)
    outputs = torch.sum(outputs,dim = 0, keepdims=  True)
    #print(outputs.shape)
    # if i == 0:
    #     outputs = outputs_single
    # else:
    #     outputs += outputs_single
    #outputs /= 5

    ret_tensor[ :,5] += torch.sum(outputs[:,5:11],dim = -1)
    ret_tensor[:,:5] += outputs[:,:5]
    ret_tensor[ :,6:] += outputs[:,11:]
ret_tensor /= (5*5)
_, preds = torch.max(ret_tensor, 1)

pred_label = preds.cpu().detach().numpy()
ret_tensor = ret_tensor.cpu().detach().numpy()
print(pred_label, ret_tensor)
results = {}
results['label']=int(pred_label[0])
results['probability']=ret_tensor[0].tolist()
print(results)
#return flask.Response(response=json.dumps(results), status=200, mimetype='text/json')