import torch.multiprocessing as mp
import os
import pickle
import ner
from ner import (
    trainer,
    active_heuristic,
    vocab,
    conlldataloader
)

from ner.tensor_logger import Logger

from ner.constants import (
    ACTIVE_LEARNING_SAMPLE,
    ACTIVE_LEARNING_TOP_K,
)

from ner.models.cached_bilstm_crf import (
    CachedBiLSTMCRF,
)

from ner.models.cached_embedder import (
    CachedEmbedder,
)

from ner.models.elmo import (
    FrozenELMo,
)

import torch
from tqdm import tqdm

from database_manager import DatabaseManager

def load(cached_embedder, session_dir):
    path = os.path.join(session_dir, "cached_embedder.pkl")
    if os.path.exists(path):
        print("loading cached embedding vectors")
        with open(os.path.join(path), 'rb') as f:
            save_state = pickle.load(f)
            cached_embedder.load(save_state, 'cached_embedder')

# def train(model):
#     # Construct data_loader, optimizer, etc.
#     for data, labels in data_loader:
#         optimizer.zero_grad()
#         loss_fn(model(data), labels).backward()
#         optimizer.step()  # This will update the shared parameters
def train(model, train_data, vocab, tag_vocab):
    trainer = ner.trainer.Trainer(
        model=model,
        learning_rate=0.01,
        weight_decay=1e-4,
        momentum=0,
        optimizer_type='SGD',
        vocab=vocab,
        tags=tag_vocab,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        train_dataset=train_data,
        logger=None,
        device='cpu',
        verbose_print=True,
        verbose_log=False,
        test_dataset=[],
        train_label_fn=lambda data, index : (data[index][0], data[index][1][0], data[index][1][1]),
        test_label_fn=lambda data, index: (data[index][0], data[index][1][0], data[index][1][1]),
        epoch_comparator=None,
    )

    train_data_loader = conlldataloader.get_data_loader(
        vocab,
        tag_vocab,
        train_data,
        1,
        False,
        0,
        label_fn=lambda data, index : (data[index][0], data[index][1][0], data[index][1][1]),
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-9, momentum=0)
    loss_sum = 0.0
    # model.train()
    print('starting epoch!!!!!!!')
    with tqdm(train_data_loader) as pbar:
        for i, (s_ids, x, x_chars, y) in enumerate(pbar):

            print(f'at iteration step 2: {i}')
            model.zero_grad()
            print(f'lets go forward!: {i}')
            model_loss = model.compute_mle(x, x_chars, y, s_ids=s_ids)
            print(f'loss iteration: {i}')
            loss = torch.mean(model_loss)
            loss.backward() # backpropogate
            # loss = torch.mean(model_loss)
            optimizer.step() # update parameters
            loss_sum += loss.item()

            # update TQDM bar
            pbar.set_postfix(
                loss_avg=loss_sum/(i + 1),
                epoch="{}/{}".format(0 + 1, 1)
            )

            pbar.refresh()
            print(f'finishing iteration: {i}')

        # model.eval()
        # model.train()
        loss_sum /= len(train_data_loader)

if __name__ == '__main__':
    db = DatabaseManager()
    db.set_session('0')
    db.load()
    vocab = db.vocab
    tag_vocab = ner.vocab.build_output_vocab([f'B-ADR', f'I-ADR', 'O'])
    configuration = db.configuration
    num_processes = 4
    cached_embedder = CachedEmbedder(
        embedder=FrozenELMo.instance(),
        embedding_dimensions=FrozenELMo.DIMENSIONS,
    )
    load(cached_embedder, 'data/0/')
    model = CachedBiLSTMCRF(
        vocab=vocab,
        tag_set=tag_vocab,
        hidden_dim=300, # configuration.get_key('model_schema/hidden_dim'),
        batch_size=1,
        embedder=cached_embedder,
    )
    # NOTE: this is required for the ``fork`` method to work
    # print(db.database.keys())
    # model.share_memory()
    train_data = [(s_id, db.database[s_id]) for s_id in range(5)]
    nt_data = []
    for s_id, (entry, label) in train_data:
        nt_data.append((s_id, (entry, ['O'] * len(entry))))
    processes = []
    for rank in range(num_processes):
        p = mp.Process(target=train, args=(model, nt_data, vocab, tag_vocab))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()