from torchtext import data
from torchtext import datasets
import spacy
import argparse
import torch
import numpy as np
import torch.nn as nn
import torch.distributions #this package provides a lot of nice abstractions for policy gradients
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--devid", type=int, default=-1)

    parser.add_argument("--model", choices=["Soft"], default="Soft")
    
    parser.add_argument("--minfreq", type=int, default=5)
    parser.add_argument("--sentlen", type=int, default=20)
    
    parser.add_argument("--nhid", type=int, default=256)
    parser.add_argument("--embdim", type=int, default=620)
    parser.add_argument("--nlayers", type=int, default=1)
    parser.add_argument("--maxout", type=int, default=1000)

    parser.add_argument("--maxnorm", type=float, default=None)
    parser.add_argument("--dropout", type=float, default=0)

    parser.add_argument("--epochs", type=int, default=10)

    parser.add_argument("--optim", choices=["SGD", "Adam"], default="Adam")

    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--lrd", type=float, default=0.25)
    parser.add_argument("--wd", type=float, default=1e-4)

    parser.add_argument("--bsize", type=int, default=32)
    parser.add_argument("--bptt", type=int, default=32)
    parser.add_argument("--clip", type=float, default=5)

    # Adam parameters
    parser.add_argument("--b1", type=float, default=0.9)
    parser.add_argument("--b2", type=float, default=0.999)
    parser.add_argument("--eps", type=float, default=1e-8)

    return parser.parse_args()

args = parse_args()

#Turn off CuDNN
torch.manual_seed(1111)
if args.devid >= 0:
    torch.cuda.manual_seed_all(1111)
    torch.backends.cudnn.enabled = False
    print("Cudnn is enabled: {}".format(torch.backends.cudnn.enabled))

#Setup Dataset
spacy_de = spacy.load('de')
spacy_en = spacy.load('en')

def tokenize_de(text):
    return [tok.text for tok in spacy_de.tokenizer(text)]

def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]

#Add beginning and end tokens to target sentences
BOS_WORD = '<s>'
EOS_WORD = '</s>'
DE = data.Field(tokenize=tokenize_de)
EN = data.Field(tokenize=tokenize_en, init_token = BOS_WORD, eos_token = EOS_WORD) # only target needs BOS/EOS

train, val, test = datasets.IWSLT.splits(exts=('.de', '.en'), fields=(DE, EN), 
                                         filter_pred=lambda x: len(vars(x)['src']) <= args.sentlen and 
                                         len(vars(x)['trg']) <= args.sentlen)

#Replace tokens that appear less than minfreq times as <unk>
DE.build_vocab(train.src, min_freq=args.minfreq)
EN.build_vocab(train.trg, min_freq=args.minfreq)

#Find pad token
padidx_en = EN.vocab.stoi["<pad>"]

train_iter, val_iter = data.BucketIterator.splits((train, val), batch_size=args.bsize, device=args.devid,
                                                  repeat=False, sort_key=lambda x: len(x.src))


class AttnNetwork(nn.Module):
    def __init__(self, vocab_size_de, vocab_size_en, word_dim=args.embdim, hidden_dim=args.nhid, n_layers=args.nlayers, maxout=args.maxout):
        super(AttnNetwork, self).__init__()
        self.hidden_dim = hidden_dim
        self.encoder = nn.LSTM(word_dim, hidden_dim, num_layers = args.nlayers, batch_first = True)
        self.decoder = nn.LSTM(word_dim, hidden_dim, num_layers = args.nlayers, batch_first = True)
        self.embedding_de = nn.Embedding(vocab_size_de, word_dim)
        self.embedding_en = nn.Embedding(vocab_size_en, word_dim)
        
        #Transformation on s_t-1
        self.linear_U = nn.Linear(hidden_dim, 2*maxout)
        
        #Transformation on y_i-1
        self.linear_V = nn.Linear(word_dim, 2*maxout)
        
        #Transformation on context vector 
        #This layer needs to be x2 input when using bidirectional LSTM
        self.linear_C = nn.Linear(hidden_dim, 2*maxout)
        
        #Maxout hidden layer
        self.linear_W(maxout, vocab_size_en)
        
        #self.vocab_layer = nn.Sequential(nn.Linear(hidden_dim*2, hidden_dim),
        #                                 nn.Tanh(), nn.Linear(hidden_dim, vocab_size_en), nn.LogSoftmax())
        
        #Need to Test dimensions...
        self.logsoftmax = nn.LogSoftmax()
        
    def forward(self, x, y, criterion, attn_type=args.model):
        emb_en = self.embedding_de(x) # Bsize x Sent Len x Emb Size 
        emb_de = self.embedding_en(y)
        h0 = Variable(torch.zeros(1, x.size(0), self.hidden_dim)) #1 x Bsize x Hidden Dim
        c0 = Variable(torch.zeros(1, x.size(0), self.hidden_dim))

        enc_h, _ = self.encoder(emb_de, (h0, c0)) #10x4x300; bsize x sent len x hidden
        dec_h, _ = self.decoder(emb_en[:, :-1], (h0, c0)) #10x3x300; bsize x sent len -1  x hidden 
      
        scores = torch.bmm(enc_h, dec_h.transpose(1,2)) #this will be a batch x source_len x target_len (10x4x3)


        loss = 0
        avg_reward = 0        
        for t in range(dec_h.size(1)):            
            attn_dist = F.softmax(scores[:, :, t], dim=1) #get attention score
            context = torch.bmm(attn_dist.unsqueeze(1), enc_h).squeeze(1)
            #pred = self.vocab_layer(torch.cat([dec_h[:, t], context], 1)) #10x300 + 10x300 -> 10x600 -> 10x300 -> Tanh -> 10xvocab size, 10x50
            label = y[:, t+1] #this will be our label
            
            #Deep output with a single maxout layer
            t = self.linear_U(dec_h[:, t]) + self.linear_V(label) + self.linear_C(context)
            t, _ = torch.max(t.view(-1, 2), 1)
            pred = self.logsoftmax(self.linear_W(t))
            
            reward = -1 * criterion(pred, label)
            avg_reward += reward.data[0]
            #reward = torch.gather(pred, 1, y.unsqueeze(1))  #our reward is log prob at the word level
            #avg_reward += reward.data.mean()                              
            loss -= reward.mean()       
        avg_reward = avg_reward/dec_h.size(1)
        return loss
    
    #predict with greedy decoding
    def predict(self, x, attn_type = args.model):
        
        emb = self.embedding(x)
        h = Variable(torch.zeros(1, x.size(0), self.hidden_dim))
        c = Variable(torch.zeros(1, x.size(0), self.hidden_dim))
        enc_h, _ = self.encoder(emb, (h, c))
        y = [Variable(torch.zeros(x.size(0)).long())]
        self.attn = []        
        for t in range(x.size(1)):
            emb_t = self.embedding(y[-1])
            dec_h, (h, c) = self.decoder(emb_t.unsqueeze(1), (h, c))
            scores = torch.bmm(enc_h, dec_h.transpose(1,2)).squeeze(2)
            attn_dist = F.softmax(scores, dim = 1)
            self.attn.append(attn_dist.data)        
            context = torch.bmm(attn_dist.unsqueeze(1), enc_h).squeeze(1)
            pred = self.vocab_layer(torch.cat([dec_h.squeeze(1), context], 1))
            _, next_token = pred.max(1)
            y.append(next_token)
        self.attn = torch.stack(self.attn, 0).transpose(0, 1)
        return torch.stack(y, 0).transpose(0, 1)
    
    #predict with beam search
    def predict_beam(self, x, attn_type = args.model):
        pass

def train(train_iter, model, criterion, optim):
    model.train()
    total_loss = 0
    for batch in tqdm(train_iter):
        x = batch.src.transpose(0, 1)
        y = batch.trg.transpose(0, 1)
        optim.zero_grad()
        bloss = model.forward(x, y, criterion)
        #correct = torch.sum(y_pred.data[:, 1:] == y.data[:, 1:]) #exclude <s> token in acc calculation    
        bloss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), 1)
        optim.step()        
        total_loss += bloss.data[0]
    return total_loss
        
def validate(val_iter, model, criterion, optim):
    model.eval()
    total_loss = 0
    for batch in val_iter:
        x = batch.src.transpose(0, 1)
        y = batch.trg.transpose(0, 1)
        bloss = model.forward(x, y, criterion)
   
        total_loss += bloss.data[0]
    return total_loss
        
if __name__ == "__main__":
    model = AttnNetwork(len(DE.vocab), len(EN.vocab))
    if args.devid >= 0:
        model.cuda(args.devid)
    
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
    
    schedule = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=1, factor=args.lrd, threshold=1e-3)
    
    # We do not want to give the model credit for predicting padding symbols,
    weight = torch.FloatTensor(len(EN.vocab)).fill_(1)
    weight[padidx_en] = 0
    if args.devid >= 0:
        weight = weight.cuda(args.devid)
    criterion = nn.NLLLoss(weight=Variable(weight), size_average=True)

    print()
    print("TRAINING:")
    for i in range(args.epochs):
        print("Epoch {}".format(i))
        train_loss = train(train_iter, model, criterion, optimizer)
        valid_loss = validate(val_iter, model, criterion, optimizer)
        schedule.step(valid_loss)
        raise Exception()
        #print("Training: {} Validation: {}".format(math.exp(train_loss/train_num.data[0]), math.exp(valid_loss/val_num.data[0])))    

    print()
    print("TESTING:")
    test_loss, test_num= validate(model, criterion, optimizer, val_iter)
    #print("Test: {}".format(math.exp(test_loss/test_num.data[0])))


    torch.save(model, 'model_attn.pt')
    
    
