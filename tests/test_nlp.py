# tests/test_nlp.py

import unittest
from src.nlp.query_processor import ArgoQueryProcessor

class TestNLP(unittest.TestCase):
    def test_processor_init(self):
        processor = ArgoQueryProcessor()
        self.assertIsNotNone(processor)

if __name__ == '__main__':
    unittest.main()