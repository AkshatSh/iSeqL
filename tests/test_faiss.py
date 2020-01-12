import unittest
import numpy as np
import faiss

'''
Converts the tutorial: https://github.com/facebookresearch/faiss/wiki/Getting-started
for FAISS (Facebook AI Similarity Search) to a unittest to ensure FAISS is set up correctly
on the machine.
'''

DIMENSIONS = 64
DATABASE_SIZE = 100000
NUM_QUERIES = 10000
FIXED_SEED = 1234
NUM_NEIGHBORS = 4
SMALL_NUM_QUERIES = 5
NORMALIZATION_FACTOR = 1000.

class TestFAISS(unittest.TestCase):
    def _test_index_results(
        self,
        query_size: int,
        database_size: int,
        neighbors: int,
        distances: np.ndarray,
        index: np.ndarray,
    ):
        # check that all the indexes are valid numbers

        def _check_index(num: int) -> bool:
            return (
                (type(num) == int or num.is_integer()) and
                num >= 0 and
                num < database_size
            )

        assert all (
            _check_index(num) for num in index.flatten().data
        )

        # assert each row of distances is sorted
        assert all(
            all(
                distances[q][i] < distances[q][i + 1]
                for i in range(distances.shape[1] - 1)
            )
            for q in range(distances.shape[0])
        )

        assert distances.shape == (query_size, neighbors)
        assert index.shape == (query_size, neighbors)

    def test_tutorial(self):
        np.random.seed(FIXED_SEED)             # make reproducible

        database = np.random.random((DATABASE_SIZE, DIMENSIONS)).astype('float32')
        database[:, 0] += np.arange(DATABASE_SIZE) / NORMALIZATION_FACTOR

        query_array = np.random.random((NUM_QUERIES, DIMENSIONS)).astype('float32')
        query_array[:, 0] += np.arange(NUM_QUERIES) / NORMALIZATION_FACTOR

        index = faiss.IndexFlatL2(DIMENSIONS)   # build the index
        assert index.is_trained # should be true
        index.add(database)                  # add vectors to the index

        assert index.ntotal == DATABASE_SIZE # assert the index size is the same as database size

        D, I = index.search(database[:SMALL_NUM_QUERIES], NUM_NEIGHBORS) # sanity check

        
        assert (I[:SMALL_NUM_QUERIES, 0] == np.arange(SMALL_NUM_QUERIES)).all()

        self._test_index_results(
            SMALL_NUM_QUERIES,
            DATABASE_SIZE,
            NUM_NEIGHBORS,
            D,
            I,
        )

        D, I = index.search(query_array, NUM_NEIGHBORS)     # actual search

        self._test_index_results(
            NUM_QUERIES,
            DATABASE_SIZE,
            NUM_NEIGHBORS,
            D,
            I,
        )
        
        