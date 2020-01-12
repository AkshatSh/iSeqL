import vocab
import constants 
import conlldataloader
import pickle
import argparse
import os
import models.bilstm_crf as bilstm_crf
import torch
from train_args import get_arg_parser
from tensor_logger import Logger
from train_conll import load_all
from tqdm import trange, tqdm
import utils


def main(args):
    train_dataset, valid_dataset, vocab, tag_vocab = load_all()
    model = bilstm_crf.BiLSTM_CRF(
        vocab, 
        tag_vocab, 
        args.embedding_dim, 
        args.hidden_dim, 
        args.batch_size
    )

    print('loading model')
    model.load_state_dict(torch.load(args.model_path, map_location=lambda storage, loc: storage))
    print('loaded model')

    data_loader = conlldataloader.get_data_loader(
        vocab, 
        tag_vocab, 
        valid_dataset, 
        1, 
        True,
        1,
    )

    correct = 0
    total = 0

    f1 = None

    bio_removed_tags = utils.remove_bio(tag_vocab.get_all())

    # iterate over epochs
    for i, (x,y) in enumerate(data_loader):
        if i > 100:
            break
        model.zero_grad()
        # print(' '.join([vocab.get_word(index) for index in x[0].data.tolist()]))
        out = model(x)

        predicted = utils.remove_bio([tag_vocab.get_word(index) for index in out[0]])
        expected = utils.remove_bio([tag_vocab.get_word(index) for index in y[0].data.tolist()])

        total += len(predicted)
        correct += sum([predicted[i] == expected[i] for i in range(len(predicted))])

        # print(predicted)
        # print(expected)
        f1_temp = utils.compute_f1(predicted, expected, bio_removed_tags)

        if f1 is None:
            f1 = f1_temp
        else:
            f1 = utils.combine_f1(f1, f1_temp)
        # break
    
    print('correct: {} total: {} accuracy: {}'.format(correct, total, correct / total))
    res = (utils.get_precision_recall_f1(f1))
    scores = []
    for key in res:
        scores.append(res[key]['f1'])
        print('{} f1: {}'.format(key, res[key]['f1']))
    avg = sum(scores) / sum([score > 0 for score in scores])
    print('Average F1: {}'.format(avg))
    # with tqdm(data_loader) as pbar:
    #     for e in trange(1):
    #         loss_sum = 0
    #         count = 0
    #         for i, (x,y) in enumerate(pbar):
    #             model.zero_grad()
    #             loss = torch.mean(model.compute_mle(x, y))
    #             # loss.backward(retain_graph=True) # backpropogate
    #             # optim.step() # update parameters
    #             loss = loss.item()
    #             loss_sum += loss
    #             count += 1

    #             # update TQDM bar
    #             pbar.set_postfix(loss=loss, loss_avg=(loss_sum / count))
    #             pbar.refresh()

    #         loss_sum /= len(data_loader)
    # print(loss_sum)
    # return loss_sum

if __name__ == "__main__":
    parser = get_arg_parser()
    args = parser.parse_args()
    main(args)
