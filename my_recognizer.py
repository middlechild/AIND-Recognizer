import warnings
from asl_data import SinglesData

import logging
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Likelihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    probabilities = []
    guesses = []

    # Implement the recognizer

    for X, lengths in test_set.get_all_Xlengths().values():
        # logging.info("X, lengths :: %s %s", X, lengths)

        best_guess = None
        best_score = float("-inf")

        logL_dict = {}

        # item == word
        for item, model in models.items():
            try:
                # Get log Likelihood for word and add to dictionary
                logL = model.score(X, lengths)
                logL_dict[item] = logL

                # Update best_guess and best_score values for word
                if logL > best_score:
                    best_guess = item
                    best_score = logL

            except:
                logL_dict[item] = float("-inf")
                pass

        # populate return lists with best values
        probabilities.append(logL_dict)
        guesses.append(best_guess)

    return probabilities, guesses

    raise NotImplementedError
