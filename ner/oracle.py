try:
    import conlldataloader
except:
    from . import conlldataloader

class Oracle(object):
    '''
    An abstract class for the Oracle interface
    '''
    def __init__(self):
        raise NotImplementedError()

    def get_label(self, inp: tuple):
        '''
        inp is in the format
        (
            index: int
            input: string list
        )

        Returns:
        (
            input_string,
            labeled_string,
        )
        '''
        raise NotImplementedError()

class SimulatedOracle(Oracle):
    '''
    Consturcts a simulated oracle that always returns the ground truth label
    '''
    def __init__(
        self, 
        dataset: conlldataloader.ConllDataSet
    ):
        self.ground_truth = dataset
    
    def get_label(self, inp):
        i, input_string = inp
        s_id, truth_sentence, truth_output = conlldataloader.default_label_fn(self.ground_truth, i)

        # sanity check to make sure 
        if (input_string != truth_sentence):
            print('{} vs {}'.format(input_string, truth_sentence))
            assert(False)

        return s_id, input_string, truth_output
    