'''
Created on Sep 2, 2018

Universidade de Sao Paulo - USP São Carlos
Instituto de Ciencias Matematicas e de Computacao
SCC5809: Redes Neurais

Exercise 1: Neural network with only one perceptron.
@author: Damares Resende
'''

import unittest
import numpy as np
import pandas as pd
from math import floor
from SCC5809.ex1.adaline import Adaline, AData


class AdalineTests(unittest.TestCase):

    def test_training_weight_update(self):
        X = pd.DataFrame(
            np.array([[1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]]))
        y = pd.DataFrame([0, 0, 0, 1])
        l_rate = 0.5
        n_epochs = 1

        model = Adaline(l_rate, n_epochs)
        model.train(X, y)

        self.assertTrue((model.weight == [0.5, 0.5, 0.5]).all())

    def test_training_end(self):
        X = pd.DataFrame(
            np.array([[1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]]))
        y = pd.DataFrame([0, 0, 0, 1])
        l_rate = 0.5
        n_epochs = 10

        model = Adaline(l_rate, n_epochs)
        model.train(X, y)

        self.assertTrue((model.weight == [-1.0, 1.0, 0.5]).all())
        self.assertTrue(model.error[-1] == 0.0)

    def test_create_noisy_instance(self):
        def test(template, noise_rate):
            noisy_instance = data._create_noisy_instance(template, noise_rate)
            n_rows, n_cols = np.shape(template)
            n_changes = floor(noise_rate * n_rows * n_cols)

            self.assertFalse((noisy_instance == template).all())
            self.assertTrue(n_rows * n_cols -
                            sum(sum(noisy_instance == template)) <= n_changes)

        data = AData()
        test(data.A_UPSIDE_DOWN, 0.7)
        test(data.A_NORMAL, 0.3)

    def test_dataset_splits_proportion(self):
        data = AData()
        data.create_dataset(60, 0.3)

        model = Adaline(0.5, 20)
        labels = pd.DataFrame(data.labels)
        dataset = pd.DataFrame([instance.reshape(-1)
                                for instance in data.dataset])
        X_train, X_test, y_train, y_test = model.create_dataset_splits(
            dataset, labels, 0.4)

        self.assertEquals(X_train.shape[0], 60 - 0.4 * 60)
        self.assertEquals(len(y_train), 60 - 0.4 * 60)
        self.assertEquals(X_test.shape[0], 0.4 * 60)
        self.assertEquals(len(y_test), 0.4 * 60)

    def test_dataset_splits_label_consistency(self):
        data = AData()
        data.create_dataset(100, 0.3)

        model = Adaline(0.5, 20)
        labels = pd.DataFrame(data.labels)
        dataset = pd.DataFrame([instance.reshape(-1)
                                for instance in data.dataset])
        X_train, X_test, y_train, y_test = model.create_dataset_splits(
            dataset, labels, 0.4)

        for idx, instance in enumerate(data.dataset):
            try:
                self.assertTrue(
                    (X_train.ix[idx] == instance.reshape(-1)).all())
                self.assertEqual(data.labels[idx], y_train.ix[idx][0])
            except KeyError:
                self.assertTrue(
                    (X_test.ix[idx] == instance.reshape(-1)).all())
                self.assertEqual(data.labels[idx], y_test.ix[idx][0])

    def test_crossvalind_unknown_type(self):

        def predict(result):
            return 1 if result > 0 else -1

        data = AData()
        data.create_dataset(100, 0.3)

        model = Adaline(0.5, 20, predict)
        labels = pd.DataFrame(data.labels)
        dataset = pd.DataFrame([instance.reshape(-1)
                                for instance in data.dataset])
        X_train, _, y_train, _ = model.create_dataset_splits(
            dataset, labels, 0.4)

        with self.assertRaises(ValueError) as context:
            model.crossvalind('dasdas', 10, X_train, y_train)
            self.assertTrue('Invalid type: dasdas' in context.exception)

    def test_crossvalind_kfold(self):
        def predict(result):
            return 1 if result > 0 else -1

        data = AData()
        data.create_dataset(100, 0.3)

        model = Adaline(0.5, 20, predict)
        labels = pd.DataFrame(data.labels)
        dataset = pd.DataFrame([instance.reshape(-1)
                                for instance in data.dataset])
        X_train, _, y_train, _ = model.create_dataset_splits(
            dataset, labels, 0.4)
        model.crossvalind('Kfold', 10, X_train, y_train)
        print('ok')


if __name__ == '__main__':
    unittest.main()
