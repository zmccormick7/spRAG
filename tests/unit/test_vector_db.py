import numpy as np
import os
import sys
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from sprag.vector_db import BasicVectorDB, VectorDB


class TestVectorDB(unittest.TestCase):
    def setUp(self):
        self.storage_directory = '/tmp'
        self.kb_id = 'test_db'
        return super().setUp()

    def tearDown(self):
        storage_path = os.path.join(self.storage_directory, 'vector_storage', f'{self.kb_id}.pkl')
        if os.path.exists(storage_path):
            os.remove(storage_path)
        return super().tearDown()

    def test__add_vectors_and_search(self):
        db = BasicVectorDB(self.kb_id, self.storage_directory)
        vectors = [np.array([1, 0]), np.array([0, 1])]
        metadata = [{'doc_id': '1', 'chunk_index': 0, 'chunk_header': 'Header1', 'chunk_text': 'Text1'},
                    {'doc_id': '2', 'chunk_index': 1, 'chunk_header': 'Header2', 'chunk_text': 'Text2'}]
        
        db.add_vectors(vectors, metadata)
        query_vector = np.array([1, 0])
        results = db.search(query_vector, top_k=1)      

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['metadata']['doc_id'], '1')
        self.assertGreaterEqual(results[0]['similarity'], 0.99)

    def test__remove_document(self):
        db = BasicVectorDB(self.kb_id, self.storage_directory)
        vectors = [np.array([1, 0]), np.array([0, 1])]
        metadata = [{'doc_id': '1', 'chunk_index': 0, 'chunk_header': 'Header1', 'chunk_text': 'Text1'},
                    {'doc_id': '2', 'chunk_index': 1, 'chunk_header': 'Header2', 'chunk_text': 'Text2'}]
        
        db.add_vectors(vectors, metadata)
        db.remove_document('1')
        
        print(db.metadata)
        self.assertEqual(len(db.metadata), 1)
        self.assertEqual(db.metadata[0]['doc_id'], '2')

    def test__empty_search(self):
        db = BasicVectorDB(self.kb_id, self.storage_directory)
        query_vector = np.array([1, 0])
        results = db.search(query_vector)
        
        self.assertEqual(len(results), 0)

    def test__save_and_load(self):
        db = BasicVectorDB(self.kb_id, self.storage_directory)
        vectors = [np.array([1, 0]), np.array([0, 1])]
        metadata = [{'doc_id': '1', 'chunk_index': 0, 'chunk_header': 'Header1', 'chunk_text': 'Text1'},
                    {'doc_id': '2', 'chunk_index': 1, 'chunk_header': 'Header2', 'chunk_text': 'Text2'}]
        
        db.add_vectors(vectors, metadata)
        db.save()
        
        new_db = BasicVectorDB(self.kb_id, self.storage_directory)
        new_db.load()
        
        self.assertEqual(len(new_db.metadata), 2)
        self.assertEqual(new_db.metadata[0]['doc_id'], '1')
        self.assertEqual(new_db.metadata[1]['doc_id'], '2')

    def test__load_from_dict(self):
        config = {
            'subclass_name': 'BasicVectorDB',
            'kb_id': 'test_db',
            'storage_directory': '/tmp'
        }
        vector_db_instance = VectorDB.from_dict(config)
        self.assertIsInstance(vector_db_instance, BasicVectorDB)
        self.assertEqual(vector_db_instance.kb_id, 'test_db')

    def test__save_and_load_from_dict(self):
        db = BasicVectorDB(self.kb_id, self.storage_directory)
        config = db.to_dict()
        vector_db_instance = VectorDB.from_dict(config)
        self.assertIsInstance(vector_db_instance, BasicVectorDB)
        self.assertEqual(vector_db_instance.kb_id, 'test_db')

    def test__assertion_error_on_mismatched_input_lengths(self):
        db = BasicVectorDB(self.kb_id, self.storage_directory)
        vectors = [np.array([1, 0])]
        metadata = [{'doc_id': '1', 'chunk_index': 0, 'chunk_header': 'Header1', 'chunk_text': 'Text1'},
                    {'doc_id': '2', 'chunk_index': 1, 'chunk_header': 'Header2', 'chunk_text': 'Text2'}]
        
        with self.assertRaises(ValueError) as context:
            db.add_vectors(vectors, metadata)
        self.assertTrue('Error in add_vectors: the number of vectors and metadata items must be the same.' in str(context.exception))

    def test__faiss_search(self):
        db = BasicVectorDB(self.kb_id, self.storage_directory, use_faiss=True)
        vectors = [np.array([1, 0]), np.array([0, 1])]
        metadata = [{'doc_id': '1', 'chunk_index': 0, 'chunk_header': 'Header1', 'chunk_text': 'Text1'},
                    {'doc_id': '2', 'chunk_index': 1, 'chunk_header': 'Header2', 'chunk_text': 'Text2'}]
        
        db.add_vectors(vectors, metadata)
        query_vector = np.array([1, 0])
        
        faiss_results = db.search(query_vector, top_k=1)

        db.use_faiss = False
        non_faiss_results = db.search(query_vector, top_k=1)

        self.assertEqual(faiss_results, non_faiss_results)


if __name__ == '__main__':
    unittest.main()