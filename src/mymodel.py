from torch import nn
import torch
class ClassificationModel(nn.Module):
	def __init__(self,n_classes = 6,temp = 1,nheads = 4):
		super(ClassificationModel, self).__init__()
		self.dropout = 0.1
		#self.batch_size = batch_size
		self.temp = temp
		self.feature_extractor = torch.hub.load('harritaylor/torchvggish', 'vggish')
		self.feature_extractor.postprocess = False
		self.feature_extractor.preprocess = False
		self.attnpooling = nn.MultiheadAttention(128, nheads, dropout=self.dropout)
		self.classifier = nn.Linear(128,n_classes)
		self.softmax = nn.Softmax(dim = -1)
		#print(hid.shape)
	def forward(self,mel):
		bs = mel.shape[0]
		mel = mel.reshape(-1,*mel.shape[2:])
		out = self.feature_extractor(mel)
		out = out.reshape(bs, -1,128)
		out = out.transpose(0, 1) # (seq, batch, hid)
		#print(out.shape)
		out = self.attnpooling(out, out, out)[0]
		#print(out.shape)
		out = out.mean(dim=0)

		out = self.classifier(out)
		out = out / self.temp
		out = self.softmax(out)
		return out

class ClassificationModelNoSoftmax(nn.Module):
	def __init__(self,n_classes = 6,temp = 1,nheads = 4):
		super(ClassificationModelNoSoftmax, self).__init__()
		self.dropout = 0.1
		#self.batch_size = batch_size
		self.temp = temp
		self.feature_extractor = torch.hub.load('harritaylor/torchvggish', 'vggish')
		self.feature_extractor.postprocess = False
		self.feature_extractor.preprocess = False
		self.attnpooling = nn.MultiheadAttention(128, nheads, dropout=self.dropout)
		self.classifier = nn.Linear(128,n_classes)
		self.softmax = nn.Softmax(dim = -1)
		#print(hid.shape)
	def forward(self,mel):
		bs = mel.shape[0]
		mel = mel.reshape(-1,*mel.shape[2:])
		out = self.feature_extractor(mel)
		out = out.reshape(bs, -1,128)
		out = out.transpose(0, 1) # (seq, batch, hid)
		#print(out.shape)
		out = self.attnpooling(out, out, out)[0]
		#print(out.shape)
		out = out.mean(dim=0)

		out = self.classifier(out)
		out = out / self.temp
		#out = self.softmax(out)
		return out

class ClassificationModelExtractFeatures(nn.Module):
	def __init__(self):
		super(ClassificationModelExtractFeatures, self).__init__()
		self.dropout = 0
		#self.batch_size = batch_size
		self.feature_extractor = torch.hub.load('harritaylor/torchvggish', 'vggish')
		self.feature_extractor.postprocess = False
		self.feature_extractor.preprocess = False
		self.attnpooling = nn.MultiheadAttention(128, 4, dropout=self.dropout)
		#self.classifier = nn.Linear(128,n_classes)
		#self.softmax = nn.Softmax(dim = -1)
		#print(hid.shape)
	def forward(self,mel):
		bs = mel.shape[0]
		mel = mel.reshape(-1, *mel.shape[2:])
		out = self.feature_extractor(mel)
		out = out.reshape(bs, -1,128)
		out = out.transpose(0, 1) # (seq, batch, hid)
		#print(out.shape)
		out = self.attnpooling(out, out, out)[0]
		#print(out.shape)
		out = out.mean(dim=0)

		#out = self.classifier(out)
		#out = out / self.temp
		#out = self.softmax(out)
		return out