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
import pickle

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--devid", type=int, default=-1)

    parser.add_argument("--model", choices=["Soft"], default="Soft")
    parser.add_argument("--preprocess", choices=["On", "Off"], default="Off")
    
    parser.add_argument("--minfreq", type=int, default=5)
    parser.add_argument("--sentlen", type=int, default=20)
    
    parser.add_argument("--nhid", type=int, default=1000)
    parser.add_argument("--embdim", type=int, default=620)
    parser.add_argument("--nlayers", type=int, default=1)
    parser.add_argument("--maxout", type=int, default=1000)

    parser.add_argument("--maxnorm", type=float, default=None)
    parser.add_argument("--dropout", type=float, default=0)

    parser.add_argument("--epochs", type=int, default=10)

    parser.add_argument("--optim", choices=["SGD", "Adam"], default="Adam")

    parser.add_argument("--lr", type=float, default=0.000001)
    parser.add_argument("--rho", type=float, default=0.95)
    parser.add_argument("--lrd", type=float, default=0.25)
    parser.add_argument("--wd", type=float, default=1e-4)

    parser.add_argument("--bsize", type=int, default=32)
    parser.add_argument("--bptt", type=int, default=32)
    parser.add_argument("--clip", type=float, default=1)

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

def preprocess():
    
    print("Preprocessing Data")
    
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
    
    vocab = [DE.vocab, EN.vocab]
    
    train_iter, val_iter = data.BucketIterator.splits((train, val), batch_size=args.bsize, device=args.devid,
                                                      repeat=False, sort_key=lambda x: len(x.src))
    
    train_x = []
    train_y = []
    for batch in tqdm(train_iter):
        train_x.append(batch.src)
        train_y.append(batch.trg)
        
    val_x = []
    val_y = []
    for batch in tqdm(val_iter):
        val_x.append(batch.src)
        val_y.append(batch.trg)
    
    with open('train_x.pkl', 'wb') as output:  # Overwrites any existing file.
        pickle.dump(train_x, output, pickle.HIGHEST_PROTOCOL)
    
        
    with open('train_y.pkl', 'wb') as output:  # Overwrites any existing file.
        pickle.dump(train_y, output, pickle.HIGHEST_PROTOCOL)
    
    with open('val_x.pkl', 'wb') as output:  # Overwrites any existing file.
        pickle.dump(val_x, output, pickle.HIGHEST_PROTOCOL)
        
    with open('val_y.pkl', 'wb') as output:  # Overwrites any existing file.
        pickle.dump(val_y, output, pickle.HIGHEST_PROTOCOL)
        
    with open('vocab.pkl', 'wb') as output:  # Overwrites any existing file.
        pickle.dump(vocab, output, pickle.HIGHEST_PROTOCOL)
    
    print("Data Loaded")
    
def load_data():
    pkl_train_x = open('train_x.pkl', 'rb')
    pkl_train_y = open('train_y.pkl', 'rb')
    pkl_val_x = open('val_x.pkl', 'rb')
    pkl_val_y = open('val_y.pkl', 'rb')
    pkl_vocab = open('vocab.pkl', 'rb')

    train_x = pickle.load(pkl_train_x)
    train_y = pickle.load(pkl_train_y)
    val_x = pickle.load(pkl_val_x)
    val_y = pickle.load(pkl_val_y)
    de_vocab, en_vocab = pickle.load(pkl_vocab)
    
    pkl_train_x.close()
    pkl_train_y.close()
    pkl_val_x.close()
    pkl_val_y.close()
    pkl_vocab.close()
    
    return train_x, train_y, val_x, val_y, de_vocab, en_vocab


class AttnNetwork(nn.Module):
    def __init__(self, vocab_size_de, vocab_size_en, word_dim=args.embdim, hidden_dim=args.nhid, n_layers=args.nlayers, maxout=args.maxout, batch_size=args.bsize):
        super(AttnNetwork, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.encoder = nn.LSTM(word_dim, hidden_dim, num_layers = args.nlayers, batch_first = True)
        self.decoder = nn.LSTM(word_dim, hidden_dim, num_layers = args.nlayers, batch_first = True)
        self.embedding_de = nn.Embedding(vocab_size_de, word_dim)
        self.embedding_en = nn.Embedding(vocab_size_en, word_dim)
        
        #Additive Attention Mechanism
        self.linear_Wa = nn.Linear(hidden_dim, hidden_dim)
        self.linear_Ua = nn.Linear(hidden_dim, hidden_dim)
        self.linear_Va = nn.Linear(hidden_dim, 1)
        
        #Transformation on s_t-1
        self.linear_U = nn.Linear(hidden_dim, 2*maxout)
        
        #Transformation on y_i-1
        self.linear_V = nn.Linear(word_dim, 2*maxout)
        
        #Transformation on context vector 
        #This layer needs to be x2 input when using bidirectional LSTM
        self.linear_C = nn.Linear(hidden_dim, 2*maxout)
        
        #Maxout hidden layer
        self.linear_W = nn.Linear(maxout, vocab_size_en)
        
        
        self.logsoftmax = nn.LogSoftmax(dim=1)
        
    def forward(self, x, y, criterion, attn_type=args.model):
        emb_de = self.embedding_de(x) # Bsize x Sent Len x Emb Size 
        emb_en = self.embedding_en(y)
        
        h0 = Variable(torch.zeros(1, x.size(0), self.hidden_dim).type_as(emb_de.data)) #1 x Bsize x Hidden Dim
        c0 = Variable(torch.zeros(1, x.size(0), self.hidden_dim).type_as(emb_de.data))
        enc_h, _ = self.encoder(emb_de, (h0, c0)) #32x16x1000; bsize x sent len x hidden
        dec_h, _ = self.decoder(emb_en[:, :-1], (h0, c0)) #32x21x1000; bsize x sent len -1  x hidden 
      
        #scores = torch.bmm(enc_h, dec_h.transpose(1,2)) #this will be a batch x source_len x target_len (32x16x21)
        

        Wa = self.linear_Wa(dec_h)
        Ua = self.linear_Ua(enc_h)
        
        scores = Variable(torch.zeros(x.size(0), Ua.size(1), Wa.size(1), self.hidden_dim).type_as(emb_de.data))
        for b in range(Wa.size(0)):
            for i in range(Ua.size(1)):
                for j in range(Wa.size(1)):
                    scores[b, i, j, :] = (Wa[b, j, :] + Ua[b, i, :]).data[0]

        scores = self.linear_Va(F.tanh(scores)).squeeze(3)
        
        loss = 0     
        for t in range(dec_h.size(1)):            
            attn_dist = F.softmax(scores[:, :, t], dim=1) #get attention score
            context = torch.bmm(attn_dist.unsqueeze(1), enc_h).squeeze(1) #batch x hidden (32x1000)
            label = y[:, t+1] #start with one word forward since first word is start of sent token
            
            #Deep output with a single maxout layer
            #(batchx2*maxout)
            #Note linear_V takes y_i-1
            t = self.linear_U(dec_h[:, t]) + self.linear_V(self.embedding_en(y[:, t])) + self.linear_C(context)
            t, _ = torch.max(t.view(x.size(0), -1, 2), 2)
            
            pred = self.logsoftmax(self.linear_W(t))
            
            reward = criterion(pred, label)
            
            loss += reward                               
        return loss
    
    #predict with greedy decoding
    def predict(self, x, attn_type = args.model):
        
        emb_de = self.embedding_de(x)
        h = Variable(torch.zeros(1, x.size(0), self.hidden_dim).type_as(emb_de.data))
        c = Variable(torch.zeros(1, x.size(0), self.hidden_dim).type_as(emb_de.data))
        enc_h, _ = self.encoder(emb_de, (h, c))
        y = [Variable(torch.zeros(x.size(0)).long())]
        self.attn = []        
        for t in range(x.size(1)):
            emb_t = self.embedding_en(y[-1])
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
    def predict_beam(self, x, padidx_s, beam_size=5):
        emb_de = self.embedding_de(x)
        h = Variable(torch.zeros(1, x.size(0), self.hidden_dim).type_as(emb_de.data))
        c = Variable(torch.zeros(1, x.size(0), self.hidden_dim).type_as(emb_de.data))
        enc_h, _ = self.encoder(emb_de, (h, c))
        
        #initialize y with start of sentence token
        y = [Variable(torch.zeros(x.size(0)).long().add_(padidx_s).type_as(emb_de.data))]

        self.attn = []    
        for t in range(x.size(1)):
            emb_t = self.embedding_en(y[-1])
            dec_h, (h, c) = self.decoder(emb_t.unsqueeze(1), (h, c))
            scores = torch.bmm(enc_h, dec_h.transpose(1,2)).squeeze(2)
            attn_dist = F.softmax(scores, dim = 1)
            context = torch.bmm(attn_dist.unsqueeze(1), enc_h).squeeze(1)
            
            label = y[:, t+1] #start with one word forward since first word is start of sent token
            
            #Deep output with a single maxout layer
            #(batchx2*maxout)
            t = self.linear_U(dec_h[:, t]) + self.linear_V(self.embedding_en(label)) + self.linear_C(context)
            t, _ = torch.max(t.view(x.size(0), -1, 2), 2)
            
            pred = self.logsoftmax(self.linear_W(t))
            
        pass

def train(train_iter, model, criterion, optimizer):
    model.train()
    total_loss = 0
    total_batch = 0
    for x, y in tqdm(train_iter):
        x = x.transpose(0, 1)
        y = y.transpose(0, 1)
        optimizer.zero_grad()
        bloss = model.forward(x, y, criterion)   
        bloss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
        optimizer.step()        
        total_loss += bloss.data[0]
        total_batch += 1
    return total_loss/total_batch
        
def validate(val_iter, model, criterion):
    model.eval()
    total_loss = 0
    total_batch = 0
    for x, y in val_iter:
        x = x.transpose(0, 1)
        y = y.transpose(0, 1)
        bloss = model.forward(x, y, criterion)
   
        total_loss += bloss.data[0]
        total_batch += 1
    return total_loss/total_batch
        
if __name__ == "__main__":
    if args.preprocess == "On":
        preprocess()
    
    train_x, train_y, val_x, val_y, de_vocab, en_vocab = load_data()

    
    model = AttnNetwork(len(de_vocab), len(en_vocab))
    
    if args.devid >= 0:
        model.cuda(args.devid)
    
    optimizer = torch.optim.Adadelta(model.parameters(), rho=args.rho)
    
#    schedule = optim.lr_scheduler.ReduceLROnPlateau(
#        optimizer, patience=1, factor=args.lrd, threshold=1e-3)
    padidx_s = en_vocab.stoi["<s>"]
    #model.predict_beam(train_x[0], padidx_s)

    # We do not want to give the model credit for predicting padding symbols,
    #Find pad token
    padidx_en = en_vocab.stoi["<pad>"]
    weight = torch.FloatTensor(len(en_vocab)).fill_(1)
    weight[padidx_en] = 0
    if args.devid >= 0:
        weight = weight.cuda(args.devid)
    criterion = nn.NLLLoss(weight=Variable(weight), size_average=True)

    print()
    print("TRAINING:")
    for i in range(args.epochs):
        train_iter = zip(train_x, train_y)
        val_iter = zip(val_x, val_y)
        print("Epoch {}".format(i))
        train_loss = train(train_iter, model, criterion, optimizer)
        valid_loss = validate(val_iter, model, criterion)
        #schedule.step(valid_loss)
        print("Training: {} Validation: {}".format(train_loss, valid_loss))
        #print("Training: {} Validation: {}".format(math.exp(train_loss/train_num.data[0]), math.exp(valid_loss/val_num.data[0])))    

    print()
    print("TESTING:")
    val_iter = zip(val_x, val_y)
    test_loss = validate(val_iter, model, criterion)
    print("Test: {}".format(test_loss))


    torch.save(model, 'model_attn.pt')
    
    
