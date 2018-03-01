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
import math
from numpy import inf
import sandbox_nmt

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--devid", type=int, default=-1)

    parser.add_argument("--model", choices=["Soft"], default="Soft")
    parser.add_argument("--preprocess", choices=["On", "Off"], default="Off")
    
    parser.add_argument("--minfreq", type=int, default=5)
    parser.add_argument("--sentlen", type=int, default=20)
    
    parser.add_argument("--bidirectional_encoder", choices=["On", "Off"], default="On")
    parser.add_argument("--nhid", type=int, default=200)
    parser.add_argument("--adddim", type=int, default=512)
    parser.add_argument("--embdim", type=int, default=200)
    parser.add_argument("--nlayers_enc", type=int, default=1)
    parser.add_argument("--nlayers_dec", type=int, default=1)
    parser.add_argument("--maxout", type=int, default=1000)

    parser.add_argument("--maxnorm", type=float, default=None)
    parser.add_argument("--dropout", type=float, default=0.3)

    parser.add_argument("--epochs", type=int, default=13)

    parser.add_argument("--optim", choices=["Adadelta", "Adam", "SGD"], default="SGD")

    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--rho", type=float, default=0.95)
    parser.add_argument("--lrd", type=float, default=0.5)
    parser.add_argument("--wd", type=float, default=1e-4)

    parser.add_argument("--bsize", type=int, default=64)
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
    
padidx_en = -1

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
    
    #process source_test data
    test_x = []
    with open('source_test.txt') as fp:
        for line in fp:
            line_vec = DE.process(line, device=args.devid, train=False)
            print(line_vec)
            test_x.append(line_vec)

#    for line in open('source_test.txt'):
#        #line.rstrip('\n')
#        line_vec = DE.process([line], device=args.devid, train=False)
#        print(line_vec)
#        #print(line_vec.src)
#        test_x.append(line_vec)
    
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
        
    with open('test_x.pkl', 'wb') as output:  # Overwrites any existing file.
        pickle.dump(test_x, output, pickle.HIGHEST_PROTOCOL)
    
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
    def __init__(self, vocab_size_de, vocab_size_en, dropout=0, word_dim=args.embdim, hidden_dim=args.nhid, n_layers_enc=args.nlayers_enc, n_layers_dec=args.nlayers_dec,
                 maxout=args.maxout, batch_size=args.bsize, add_dim=args.adddim, bidirect=args.bidirectional_encoder):
        super(AttnNetwork, self).__init__()
        self.attn = []
        self.bidirect = 1
        self.bidirect_toggle = False
        self.nlayers_enc = n_layers_enc
        self.nlayers_dec = n_layers_dec
        if bidirect == "On":
            self.bidirect = 2
            self.bidirect_toggle = True
        self.hidden_dim = hidden_dim
        self.add_dim = add_dim
        
        self.drop =nn.Dropout(dropout)
        
        self.encoder = nn.LSTM(word_dim, hidden_dim, num_layers = n_layers_enc, batch_first = True, bidirectional = self.bidirect_toggle, dropout = dropout)
        self.decoder = nn.LSTM(word_dim, hidden_dim, num_layers = n_layers_dec, batch_first = True, dropout = dropout)
        self.embedding_de = nn.Embedding(vocab_size_de, word_dim)
        self.embedding_en = nn.Embedding(vocab_size_en, word_dim)
        
        #Transformations on multi score
        self.linear_E = nn.Linear(hidden_dim * self.bidirect, hidden_dim)
        self.linear_D = nn.Linear(hidden_dim, hidden_dim)
        
        #Transformations on add score
#        self.linear_Wa = nn.Linear(hidden_dim, add_dim)
#        self.linear_Ua = nn.Linear(hidden_dim * self.bidirect, add_dim)
#        self.linear_Va = nn.Linear(add_dim, 1)
        
        
        #Transformation on context vector 
        self.linear_A = nn.Linear(3 * hidden_dim, hidden_dim)
        self.linear_C = nn.Linear(hidden_dim, vocab_size_en)
        
        self.logsoftmax = nn.LogSoftmax(dim=1)
        
    def forward(self, x, y, criterion, attn_type=args.model, visualize="Off"):
        
        

        emb_de = self.embedding_de(x) # Bsize x Sent Len x Emb Size 
        emb_en = self.embedding_en(y)
        
        h0_enc = Variable(torch.zeros(self.nlayers_enc * self.bidirect, x.size(0), self.hidden_dim).type_as(emb_de.data)) #1 x Bsize x Hidden Dim
        c0_enc = Variable(torch.zeros(self.nlayers_enc * self.bidirect, x.size(0), self.hidden_dim).type_as(emb_de.data))
        
        enc_h, (h0_enc, c0_enc) = self.encoder(emb_de, (h0_enc, c0_enc)) #32x16x1000; bsize x sent len x hidden

        #Need to change the initialized dec hidden state to the backwards layer of bilstm later...
        h0 = h0_enc[1, :, :].unsqueeze(0)
        c0 = c0_enc[1, :, :].unsqueeze(0)
        #h0 = h0_enc
        #c0 = c0_enc
        
        dec_h, (h_dec, c_dec) = self.decoder(emb_en[:, :-1], (h0, c0)) #32x21x1000; bsize x sent len -1  x hidden 
        dec_h = self.drop(dec_h) 
            
        scores = torch.bmm(self.linear_E(enc_h), self.linear_D(dec_h).transpose(1, 2)) #this will be a batch x source_len x target_len (32x16x21)

        loss = 0  
    
        for i in range(dec_h.size(1)):            
            attn_dist = F.softmax(scores[:, :, i], dim=1) #get attention score
            
            if visualize == "On":
                self.attn.append(attn_dist.data)
                

            
            context = torch.bmm(attn_dist.unsqueeze(1), enc_h).squeeze(1) #batch x hidden (32x1000)
            
            label = y[:, i+1] #start with one word forward since first word is start of sent token
            
            
            t = F.tanh(self.linear_A(torch.cat((context, dec_h[:, i]), 1)))

            pred = self.linear_C(t)
            reward = criterion(pred, label)
            
            loss += reward    
        
        if visualize == "On":
            torch.stack(self.attn, 0).transpose(0, 1)                
        
        return loss
    
    #predict with greedy decoding
    def predict_greedy(self, x, padidx_s):
        self.eval()
        
        emb_de = self.embedding_de(x)
        h_enc = Variable(torch.zeros(self.nlayers_enc * self.bidirect, x.size(0), self.hidden_dim).type_as(emb_de.data)) #1 x Bsize x Hidden Dim
        c_enc = Variable(torch.zeros(self.nlayers_enc * self.bidirect, x.size(0), self.hidden_dim).type_as(emb_de.data))
        

        
        enc_h, (h, c) = self.encoder(emb_de, (h_enc, c_enc))
        y = [Variable(torch.zeros(x.size(0)).add_(padidx_s).type_as(emb_de.data).long())]
        
#        h = h[1, :, :].unsqueeze(0)
#        c = c[1, :, :].unsqueeze(0)
        
        for i in range(1, args.sentlen):
            emb_t = self.embedding_en(y[-1])
            dec_h, (h, c) = self.decoder(emb_t.unsqueeze(1), (h, c))
            
            #Calculate Scores
            scores = torch.bmm(self.linear_E(enc_h), self.linear_D(dec_h).transpose(1, 2)) #this will be a batch x source_len x target_len (32x16x21)


            attn_dist = F.softmax(scores, dim = 1)
            self.attn.append(attn_dist.data)        
            context = torch.bmm(attn_dist.unsqueeze(1), enc_h).squeeze(1)
            
            #Generate Prediction
            t = F.tanh(self.linear_A(torch.cat((context, dec_h[:, i]), 1)))
        
            #do I need softmax here? probably not
            pred = self.logsoftmax(self.linear_C(t)) #batchxvocab
            _, next_token = pred.max(1)
            y.append(next_token)
            
        self.attn = torch.stack(self.attn, 0).transpose(0, 1)
        return torch.stack(y, 0).transpose(0, 1).data
    
        #predict with beam search
    def predict_beam(self, x, padidx_s, eosidx, en_len,  beam_size=5):
        self.eval()
        
        start_beam = beam_size
        
        emb_de = self.embedding_de(x).unsqueeze(0)


        h_enc = Variable(torch.zeros(self.nlayers_enc * self.bidirect, 1, self.hidden_dim).type_as(emb_de.data)) #1 x Bsize x Hidden Dim
        c_enc = Variable(torch.zeros(self.nlayers_enc * self.bidirect, 1, self.hidden_dim).type_as(emb_de.data))
        
        enc_h, (h, c) = self.encoder(emb_de, (h_enc, c_enc))
    
        
        #initialize y with start of sentence token; len is + 2 to deal with begin and end sentence tokens
        start_sen_tok = Variable(torch.zeros(beam_size).add_(padidx_s).type_as(emb_de.data).long()) 
        
        #Create a 2D tensor that tracks whether a beam is completed or not (has hit <eos>)
        running_eos = torch.zeros(beam_size).type_as(emb_de.data).long()
        
        #Create 2D tensor that represents beam_size x predicted word indicies, of best current predictions
        running_beam = torch.zeros(beam_size, args.sentlen).type_as(emb_de.data).long()

        running_beam[:, 0]  = start_sen_tok.data
            
        # Find best beam_size (unique) first words
        emb_t = self.embedding_en(start_sen_tok[0])

        dec_h, (h, c) = self.decoder(emb_t.unsqueeze(1), (h, c))
        
        scores = torch.bmm(self.linear_E(enc_h), self.linear_D(dec_h).transpose(1, 2)).squeeze(2) #this will be a batch x source_len x target_len (32x16x21)
        
        attn_dist = F.softmax(scores, dim = 1)
        context = torch.bmm(attn_dist.unsqueeze(1), enc_h).squeeze(1)
    
        
        t = F.tanh(self.linear_A(torch.cat((context, dec_h[:, 0]), 1)))
        
        pred = self.logsoftmax(self.linear_C(t)) #batchxvocab
        _, predk_indices = pred.topk(beam_size, dim=1)
        
        #predk_probs will keep track of the current best probabilities for each beam
        predk_probs = torch.gather(pred, 1, predk_indices)

        running_beam[:, 1]  = predk_indices.data
              
        #Create a list that keeps track of the hidden layers for each copy of the decoder
        # Initialize all 5 beams with decoder hidden state of first layer
        running_hidden = []
        for i in range(beam_size):
            running_hidden.append((h, c))
            

        

        
        #Work on all the words after the start of sentence token
        for w in range(2, args.sentlen):

            cat_probs = torch.FloatTensor().type_as(emb_de.data)
            for i in range(beam_size):
                
                if running_eos[i] == 0:
                
                    emb_t = self.embedding_en(Variable(torch.LongTensor([running_beam[i, w-1]]).type_as(running_beam)))
                    
                    h_t, c_t = running_hidden[i]
                    dec_h, (h, c) = self.decoder(emb_t.unsqueeze(1), (h_t, c_t))
                    running_hidden[i] = (h, c)
                    
                    scores = torch.bmm(self.linear_E(enc_h), self.linear_D(dec_h).transpose(1, 2)).squeeze(2)
                    
                    attn_dist = F.softmax(scores, dim = 1)
                    context = torch.bmm(attn_dist.unsqueeze(1), enc_h).squeeze(1)
                    
                    t = F.tanh(self.linear_A(torch.cat((context, dec_h[:, 0]), 1)))
                    pred = self.logsoftmax(self.linear_C(t)) + predk_probs.squeeze(0)[i] #batchxvocab
    
                    cat_probs = torch.cat((cat_probs, pred.data), 1)
            
            if beam_size > 0:
                _, predk_indices =cat_probs.topk(beam_size, dim=1)

                gathered_probs = Variable(torch.gather(cat_probs, 1, predk_indices))
    
                while gathered_probs.size(1) < start_beam:
                    gathered_probs = torch.cat((gathered_probs, predk_probs[:, gathered_probs.size(1)].unsqueeze(0).type_as(predk_probs.data)), 1)
                
                predk_probs = gathered_probs

                current_ind = torch.remainder(predk_indices, en_len)
                while current_ind.size(1) < start_beam:
                    current_ind = torch.cat((current_ind, torch.LongTensor([[0]])), 1)
                prev_ind = torch.div(predk_indices, en_len)
                while prev_ind.size(1) < start_beam:
                    prev_ind = torch.cat((prev_ind, torch.LongTensor([[prev_ind.size(1)]])), 1)
                            
                running_beam = running_beam[prev_ind, :].squeeze(0)
                
                running_beam[:, w] = current_ind.transpose(0, 1) 
                running_eos += (torch.eq(current_ind.transpose(0, 1), eosidx).long() * w).unsqueeze(1)
                
                #move completed beams to the bottom of the list, and reduce beam size
                beam_reduce = 0
                counter = beam_size
                i = 0
                while counter > 0:
                    if running_eos[i] > 0:
                        running_beam = sandbox_nmt.move_to_bottom(running_beam, i)
                        predk_probs = sandbox_nmt.move_to_bottom(predk_probs.transpose(0, 1), i).transpose(0, 1)
                        running_eos = sandbox_nmt.move_to_bottom(running_eos.unsqueeze(1), i).squeeze(1)
                        running_hidden.append(running_hidden.pop(i))
                        beam_reduce += 1
                    else:
                        i += 1
                    counter = counter - 1
                beam_size -= beam_reduce
            

        print(running_beam)
        
        _, max_index = predk_probs.topk(start_beam,1)

        final_pred = running_beam[max_index.data.squeeze(0), :].type_as(x.data)
        
        print(final_pred)

        return final_pred
    
    def gen_predictions(DE):
        #self.eval

        
        pass
        
        
    
    

def train(train_iter, model, criterion, optimizer, train_len):
    model.train()
    total_loss = 0
    nwords = 0
    for x, y in tqdm(train_iter):
        x = x.transpose(0, 1)
        y = y.transpose(0, 1)
        optimizer.zero_grad()
        bloss = model.forward(x, y, criterion)
        total_loss += bloss.data[0]
        nwords += y.ne(padidx_en).int().sum()
        bloss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
        optimizer.step()

        #total_loss += bloss.data[0] * (x.size(0) / train_len)
    
    word_loss = total_loss/nwords.data[0]

    return word_loss

def validate(val_iter, model, criterion, val_len, startosidx, eosidx, en_vocab):
    model.eval()
    total_loss = 0
    total_bleu = 0
    nwords = 0
    for x, y in tqdm(val_iter):
        x = x.transpose(0, 1)
        y = y.transpose(0, 1)
        
        nwords += y.ne(padidx_en).int().sum()
        bloss = model.forward(x, y, criterion)
        total_loss += bloss.data[0]

    word_loss = total_loss/nwords.data[0]
    
    return word_loss, total_bleu

def escape(l):
    return l.replace("\"", "<quote>").replace(",", "<comma>")
        
if __name__ == "__main__":
    if args.preprocess == "On":
        preprocess()
    
    train_x, train_y, val_x, val_y, de_vocab, en_vocab = load_data()
    startosidx = en_vocab.stoi["<s>"]
    eosidx = en_vocab.stoi["</s>"]
    
    train_len = 0
    val_len = 0
    for i in range(len(train_x)):
        train_len += train_x[i].size(1)
    for i in range(len(val_x)):
        val_len += val_x[i].size(1)

    model = AttnNetwork(len(de_vocab), len(en_vocab), dropout=args.dropout)
    #model.load_state_dict(torch.load("model_attn_SR.pth", map_location=lambda storage, loc: storage))
    
    
#    f = open('output.txt','w')
#    f.write("id,word\n")
    
    
#    i = 1
#    with open('source_test.txt', 'r') as fp:
#        for line in tqdm(fp):
#            line_vec = Variable(torch.Tensor([de_vocab.stoi[i ]for i in line.split()])).type_as(train_x[0])
#            final_pred = model.predict_beam(line_vec, startosidx, eosidx, len(en_vocab), beam_size=100)
#            f.write("{},".format(i))
#            for j in range(final_pred.size(0)):
#                beam = final_pred[j, :]
#                beam_join = escape("|".join(en_vocab.itos[j] for j in beam[1:4].tolist()))
#                print(beam_join)
#
#                f.write("{} ".format(beam_join)) 
#            f.write("\n")
#            #raise Exception()
#            i += 1
#    f.close()
#                    
#    raise Exception()
    
    
    if args.devid >= 0:
        model.cuda(args.devid)
    if args.optim == "Adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr = args.lr)
    elif args.optim == "Adadelta":
        optimizer = torch.optim.Adadelta(model.parameters(), rho=args.rho)
    elif args.optim == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 11, 12, 13], gamma=0.5)
        #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1, factor=args.lrd, threshold=1e-3)

    # We do not want to give the model credit for predicting padding symbols,
    #Find pad token
    padidx_en = en_vocab.stoi["<pad>"]
    criterion = nn.CrossEntropyLoss(size_average=False, ignore_index=padidx_en)
    

    print()
    print("TRAINING:")
    for i in range(args.epochs):
        scheduler.step()
        
        train_iter = zip(train_x, train_y)
        val_iter = zip(val_x, val_y)
        print("Epoch {}".format(i))
        train_loss = train(train_iter, model, criterion, optimizer, train_len)
        valid_loss, bleu = validate(val_iter, model, criterion, val_len, startosidx, eosidx, en_vocab)
        #scheduler.step(valid_loss)
        
        print("Training: {} Validation: {}".format(math.exp(train_loss), math.exp(valid_loss)))

    print()
    print("TESTING:")
    val_iter = zip(val_x, val_y)
    test_loss, bleu = validate(val_iter, model, criterion, val_len, startosidx, eosidx, en_vocab)
    print("Test: {}".format(math.exp(test_loss)))


    torch.save(model.state_dict(), 'model_attn_SR.pth')
    
    
