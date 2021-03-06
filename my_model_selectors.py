import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences

import logging
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant
    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # Implement model selection based on BIC scores

        scores = []
        n_components = range(self.min_n_components, self.max_n_components+1)

        # n = number of HMM states
        # d = number of features
        # (The shape attribute for numpy arrays returns the dimensions of the array)
        n, d = self.X.shape

        for n_component in n_components:
            try:
                model = self.base_model(n_component)
                logL = model.score(self.X, self.lengths)
                p = (n_component * (n_component-1)) + (n_component-1) + (2 * d * n_component)
                logN = np.log(n)

                bic_score = -2 * logL + p * logN

                # Add bic_score to scores list
                scores.append(bic_score)

            except:
                # logging.info("except")
                pass

        # Return best model based on BIC
        states = n_components[np.argmin(scores)] if scores else self.n_constant
        return self.base_model(states)

        raise NotImplementedError


class SelectorDIC(ModelSelector):
    """ select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    """

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # Implement model selection based on DIC scores

        scores = []
        logL_all = []
        n_components = range(self.min_n_components, self.max_n_components+1)

        for n_component in n_components:
            try:
                model = self.base_model(n_component)

                # 1. Get logL values for all words
                logL = model.score(self.X, self.lengths)
                logL_all.append(logL)

            except:
                # logging.info("except")
                pass

        # 2. Implement DIC formula
        m = len(n_components)
        sum_logL_all = sum(logL_all)

        for logL in logL_all:
            # DIC = likelihood(this word) - average likelihood(other words)
            dic_score = logL - ((sum_logL_all - logL) / (m - 1))

            # Add dic_score to scores list
            scores.append(dic_score)

        # Return best model based on DIC
        states = (n_components[np.argmax(scores)] or n_components[0]) if scores else self.n_constant
        return self.base_model(states)

        raise NotImplementedError


class SelectorCV(ModelSelector):
    """ select best model based on average log Likelihood of cross-validation folds
    """

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # Implement model selection using CV

        scores = []
        split_method = KFold()
        n_components = range(self.min_n_components, self.max_n_components+1)

        for n_component in n_components:
            try:
                # Build model
                model = self.base_model(n_component)

                # Conditional statement necessary in case splitting is not possible
                if len(self.sequences) < 2:
                    # Add scores mean to scores list
                    scores.append(np.mean(model.score(self.X, self.lengths)))

                else:
                    test_scores = []

                    for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                        # Setup training sequences
                        self.X, self.lengths = combine_sequences(cv_train_idx, self.sequences)
                        # Setup testing sequences
                        test_X, test_lengths = combine_sequences(cv_test_idx, self.sequences)

                        test_scores.append(model.score(test_X, test_lengths))

                    # Add test_scores mean to scores list
                    scores.append(np.mean(test_scores))

            except:
                # logging.info("except")
                pass

        # Return best model
        states = n_components[np.argmax(scores)] if scores else self.n_constant
        return self.base_model(states)

        raise NotImplementedError
