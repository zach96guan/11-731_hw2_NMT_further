# coding=utf-8

"""
A very basic implementation of neural machine translation

Usage:
    nmt.py train --train-src=<file> --dev-src=<file> --vocab=<file> [options]
    nmt.py decode [options] MODEL_PATH TEST_SOURCE_FILE OUTPUT_FILE
    nmt.py decode [options] MODEL_PATH TEST_SOURCE_FILE TEST_TARGET_FILE OUTPUT_FILE

Options:
    -h --help                               show this screen.
    --cuda                                  use GPU
    --train-src=<file>                      train source file
    --train-tgt=<file>                      train target file
    --dev-src=<file>                        dev source file
    --dev-tgt=<file>                        dev target file
    --vocab=<file>                          vocab file
    --seed=<int>                            seed [default: 0]
    --batch-size=<int>                      batch size [default: 32]
    --embed-size=<int>                      embedding size [default: 256]
    --hidden-size=<int>                     hidden size [default: 256]
    --clip-grad=<float>                     gradient clipping [default: 5.0]
    --log-every=<int>                       log every [default: 10]
    --max-epoch=<int>                       max epoch [default: 30]
    --patience=<int>                        wait for how many iterations to decay learning rate [default: 5]
    --max-num-trial=<int>                   terminate training after how many trials [default: 5]
    --lr-decay=<float>                      learning rate decay [default: 0.5]
    --beam-size=<int>                       beam size [default: 5]
    --lr=<float>                            learning rate [default: 0.001]
    --uniform-init=<float>                  uniformly initialize all parameters [default: 0.1]
    --save-to=<file>                        model save path
    --valid-niter=<int>                     perform validation after how many iterations [default: 2000]
    --dropout=<float>                       dropout [default: 0.2]
    --max-decoding-time-step=<int>          maximum number of decoding time steps [default: 70]
"""

import math
import pickle
import sys
import time

import numpy as np
from typing import List, Tuple, Dict, Set, Union
from docopt import docopt
from tqdm import tqdm

import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F

from utils import input_transpose, read_corpus, batch_iter
#from vocab import Vocab, VocabEntry
from data import MTDataset, MTDataLoader, Vocab

pad_token = '<pad>'
pad_id = 0
sos_id = 1
eos_id = 2
unk_id = 3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, dropout):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_size, vocab_size)
        self.logSoftmax = nn.LogSoftmax(dim=2)

    def forward(self, indices, hidden):
        embedded = self.embedding(indices)
        embedded = F.relu(embedded)
        embedded = self.dropout(embedded)
        output, hidden = self.lstm(embedded, hidden)

        output = self.logSoftmax(self.out(hidden[0]))
        return output, hidden

class NMT(nn.Module):

    def __init__(self, embed_size, hidden_size, vocab, dropout_rate=0.):
        super(NMT, self).__init__()

        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.encoder = Encoder(self.vocab_size, embed_size, hidden_size, dropout_rate)
        self.criterion = nn.NLLLoss(reduction="sum", ignore_index=pad_id)
        self.device = device

        # initialize neural network layers...

    def __call__(self, sents: List[List[str]]) -> Tensor:
        return self.encode(sents)

    def encode(self, sents: List[List[str]]) -> Tensor:
        sents_t = input_transpose(sents, pad_token)
        # sent_t = [sent_len, batch_size]
        sent_len = len(sents_t)
        batch_size = len(sents_t[0])

        indices_list = [[self.vocab.to_idx(w) for w in s] for s in sents_t]
        indices = torch.LongTensor(indices_list).to(device)
        loss = 0

        hidden_init = torch.FloatTensor(1, batch_size, self.hidden_size).uniform_(-0.1, 0.1).to(device)
        cell_init = torch.FloatTensor(1, batch_size, self.hidden_size).uniform_(-0.1, 0.1).to(device)
        hidden = (hidden_init, cell_init)
        for i in range(sent_len-1):
            word_batch = indices[i,:].unsqueeze(0)
            output, hidden = self.encoder(word_batch, hidden)
            loss += self.criterion(output.squeeze(0), indices[i+1])
        # print(loss.item())
        return loss
 
    def evaluate_ppl(self, dev_data, batch_size: int=32):
        """
        Evaluate perplexity on dev sentences

        Args:
            dev_data: a list of dev sentences
            batch_size: batch size
        
        Returns:
            ppl: the perplexity on dev sentences
        """

        cum_loss = 0.
        cum_tgt_words = 0.

        # you may want to wrap the following code using a context manager provided
        # by the NN library to signal the backend to not to keep gradient information
        # e.g., `torch.no_grad()`

        with torch.no_grad():
            for sents in batch_iter(dev_data, batch_size):
                loss = self.__call__(sents)

                cum_loss += loss
                # should I include 0? 
                tgt_word_num_to_predict = sum(len(s[1:]) for s in sents)  # omitting the leading `<s>`
                cum_tgt_words += tgt_word_num_to_predict

            ppl = torch.exp(cum_loss / cum_tgt_words)

        return ppl

    @staticmethod
    def load(model_path: str):
        """
        Load a pre-trained model

        Returns:
            model: the loaded model
        """
        model = torch.load(model_path)
        model.eval()
        return model

    def save(self, path: str):
        """
        Save current model to file
        """
        torch.save(self, path)

def train(args: Dict[str, str]):
    train_data = read_corpus(args['--train-src'], source='src')
    dev_data = read_corpus(args['--dev-src'], source='src')

    train_batch_size = int(args['--batch-size'])
    train_batch_size = 64
    clip_grad = float(args['--clip-grad'])
    valid_niter = int(args['--valid-niter'])
    # valid_niter = 100
    log_every = int(args['--log-every'])
    # log_every = 1
    dropout_rate = float(args['--dropout'])
    model_save_path = args['--save-to']
    optim_save_path = "work_dir/optim.bin"

    # vocab = pickle.load(open(args['--vocab'], 'rb'))
    # initialize vocabe
    vocab = Vocab.from_data_files(args['--vocab'])
    print("vocab size = %d" % len(vocab))

    model = NMT(embed_size=int(args['--embed-size']),
                hidden_size=int(args['--hidden-size']),
                dropout_rate=dropout_rate, 
                vocab=vocab).to(device)
    #model = torch.load(model_save_path)
    lr = float(args['--lr'])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    num_trial = 0
    train_iter = patience = cum_loss = report_loss = cumulative_tgt_words = report_tgt_words = 0
    cumulative_examples = report_examples = epoch = valid_num = 0
    hist_valid_scores = []
    train_time = begin_time = time.time()
    print('begin Maximum Likelihood training')

    while True:
        epoch += 1

        for sents in batch_iter(train_data, batch_size=train_batch_size, shuffle=True):
            train_iter += 1

            batch_size = len(sents)

            # (batch_size)
            optimizer.zero_grad()
            loss = model(sents)

            report_loss += loss.item()
            cum_loss += loss.item()

            loss.backward()
            if clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()

            tgt_words_num_to_predict = sum(len(s[1:]) for s in sents)  # omitting leading `<s>`
            report_tgt_words += tgt_words_num_to_predict
            cumulative_tgt_words += tgt_words_num_to_predict
            report_examples += batch_size
            cumulative_examples += batch_size

            if train_iter % log_every == 0:
                print('epoch %d, iter %d, avg. loss %.2f, avg. ppl %.2f ' \
                      'cum. examples %d, speed %.2f words/sec, time elapsed %.2f sec' % (epoch, train_iter,
                                                                                         report_loss / report_examples,
                                                                                         math.exp(report_loss / report_tgt_words),
                                                                                         cumulative_examples,
                                                                                         report_tgt_words / (time.time() - train_time),
                                                                                         time.time() - begin_time), file=sys.stderr)

                train_time = time.time()
                report_loss = report_tgt_words = report_examples = 0.

            # the following code performs validation on dev set, and controls the learning schedule
            # if the dev score is better than the last check point, then the current model is saved.
            # otherwise, we allow for that performance degeneration for up to `--patience` times;
            # if the dev score does not increase after `--patience` iterations, we reload the previously
            # saved best model (and the state of the optimizer), halve the learning rate and continue
            # training. This repeats for up to `--max-num-trial` times.
            if train_iter % valid_niter == 0:
                print('epoch %d, iter %d, cum. loss %.2f, cum. ppl %.2f cum. examples %d' % (epoch, train_iter,
                                                                                         cum_loss / cumulative_examples,
                                                                                         math.exp(cum_loss / cumulative_tgt_words),
                                                                                         cumulative_examples), file=sys.stderr)

                cum_loss = cumulative_examples = cumulative_tgt_words = 0.
                valid_num += 1

                print('begin validation ...', file=sys.stderr)

                # compute dev. ppl and bleu
                dev_ppl = model.evaluate_ppl(dev_data, batch_size=128)   # dev batch size can be a bit larger
                valid_metric = -dev_ppl

                print('validation: iter %d, dev. ppl %f' % (train_iter, dev_ppl), file=sys.stderr)

                is_better = len(hist_valid_scores) == 0 or valid_metric > max(hist_valid_scores)
                hist_valid_scores.append(valid_metric)

                if is_better:
                    patience = 0
                    print('save currently the best model to [%s]' % model_save_path, file=sys.stderr)
                    model.save(model_save_path)
                    # You may also save the optimizer's state
                    torch.save(optimizer.state_dict(), optim_save_path)
                elif patience < int(args['--patience']):
                    patience += 1
                    print('hit patience %d' % patience, file=sys.stderr)

                    if patience == int(args['--patience']):
                        num_trial += 1
                        print('hit #%d trial' % num_trial, file=sys.stderr)
                        if num_trial == int(args['--max-num-trial']):
                            print('early stop!', file=sys.stderr)
                            exit(0)

                        # decay learning rate, and restore from previously best checkpoint
                        lr = lr * float(args['--lr-decay'])
                        print('load previously best model and decay learning rate to %f' % lr, file=sys.stderr)

                        # load model
                        model = NMT.load(args['MODEL_PATH'])

                        print('restore parameters of the optimizers', file=sys.stderr)
                        # You may also need to load the state of the optimizer saved before
                        optimizer.load_state_dict(torch.load(optim_save_path))

                        # reset patience
                        patience = 0

                if epoch == int(args['--max-epoch']):
                    print('reached maximum number of epochs!', file=sys.stderr)
                    exit(0)

def decode(args: Dict[str, str]):
    """
    performs decoding on a test set, and save the best-scoring decoding results. 
    If the target gold-standard sentences are given, the function also computes
    corpus-level BLEU score.
    """
    test_data = read_corpus(args['TEST_SOURCE_FILE'], source='src')

    print(f"load model from {args['MODEL_PATH']}", file=sys.stderr)
    model = NMT.load(args['MODEL_PATH'])
    model.encoder.dropout = nn.Dropout(0.)

    ces = []
    with torch.no_grad():
        for sent in tqdm(test_data, desc='Decoding', file=sys.stdout):
            loss = model([sent]).item()
            ce = loss / len(sent)
            ces.append(ce)

    with open(args['OUTPUT_FILE'], 'w') as f:
        for sent, ce in zip(test_data, ces):
            f.write(str(ce) + '\n')

def main():
    args = docopt(__doc__)

    # seed the random number generator (RNG), you may
    # also want to seed the RNG of tensorflow, pytorch, dynet, etc.
    seed = int(args['--seed'])
    np.random.seed(seed * 13 // 7)

    if args['train']:
        train(args)
    elif args['decode']:
        decode(args)
    else:
        raise RuntimeError(f'invalid mode')


if __name__ == '__main__':
    main()
