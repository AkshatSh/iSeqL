import torch
from torch import nn
from allennlp.modules.elmo import Elmo, batch_to_ids
from typing import (
    List
)

def load_elmo():
    '''
    load a pretrained elmo model
    '''
    with torch.no_grad():
        options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
        weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

        scalars = {
            "scalar_weights_0": {
                "scalars": [
                    0.1858515590429306,
                    0.2843584716320038,
                    -0.4623725414276123
                ],
                "normed_scalars": [
                    0.38073545694351196,
                    0.42014995217323303,
                    0.1991146206855774
                ],
                "gamma": 0.9229910373687744
            }
        }

        return Elmo(
            options_file,
            weight_file,
            1,
            scalar_mix_parameters=scalars["scalar_weights_0"]["scalars"],
            dropout=0,
            requires_grad=False,
        )

class FrozenELMo(nn.Module):

    '''
    Singleton instance for efficency, since the 
    model should be frozen, this should make no difference
    '''
    singleton_instance = None
    DIMENSIONS = 1024

    '''
    A wrapper for ELMo embeddings that ensures that the
    embedding layer is frozen and not retrained
    '''
    def __init__(self):
        super(FrozenELMo, self).__init__()
        self.elmo = load_elmo()
        # self.elmo_embedder = ElmoEmbedder(cuda_device='cpu')
    
    def forward(self, character_ids: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.elmo(character_ids)
        # with torch.no_grad():
        #     return self.elmo_embedder.embed_batch(character_ids)
    
    def get_embedding_from_sentence(
        self,
        raw_sentence: List[str],
        device: str = 'cpu',
    ) -> torch.Tensor:
        if len(raw_sentence) == 0: # pragma: no cover
            return torch.Tensor().to(device)
        character_ids = batch_to_ids([raw_sentence]).to(device)
        embeddings = self(character_ids)
        embeded_sentence = embeddings['elmo_representations'][0]

        return embeded_sentence[0] # single sentence
    
    @staticmethod
    def instance():
        '''
        Singleton fetcher for ELMo model
        '''
        if FrozenELMo.singleton_instance is None:
            FrozenELMo.singleton_instance = FrozenELMo()
        
        return FrozenELMo.singleton_instance
