# -*- coding: utf-8 -*-
"""
Link prediction

To integrate (vanilla model)
https://docs.ampligraph.org/en/2.1.0/tutorials/AmpliGraphBasicsTutorial.html


from ampligraph.latent_features import ScoringBasedEmbeddingModel
model = ScoringBasedEmbeddingModel(
    k=200,
    eta=15,
    scoring_type='ComplEx',
    seed=23)

from tensorflow.keras.optimizers import Adam
from ampligraph.latent_features.loss_functions import get as get_loss
from ampligraph.latent_features.regularizers import get as get_regularizer

optimizer = Adam(learning_rate=0.0005)
loss = get_loss('multiclass_nll')
regularizer = get_regularizer('LP', {'p':1, 'lambda':1e-5})

model.compile(loss=loss,
              optimizer=optimizer,
              entity_relation_regularizer=regularizer,
              entity_relation_initializer='glorot_uniform')


model.fit(
    X_train,
    batch_size=100,
    epochs=100,
    verbose=True
)

(and cf. code for ranking hypotheses)
"""
import itertools
from loguru import logger
import pandas as pd
import numpy as np
from scipy.special import expit
from tensorflow.keras.optimizers import Adam
from ampligraph.evaluation import train_test_split_no_unseen 
from ampligraph.latent_features import ScoringBasedEmbeddingModel
from ampligraph.latent_features.loss_functions import get as get_loss
from ampligraph.latent_features.regularizers import get as get_regularizer
from ampligraph.evaluation import mr_score, mrr_score, hits_at_n_score



class LinkPredictor:
    """ LP-Based hypothesis generation """
    def __init__(self, data: str):
        self.data = pd.read_csv(data, index_col=0)
        self.cp = 'https://data.cooperationdatabank.org/vocab/prop/'
        self.pos_e = f"{self.cp}hasPositiveEffectOn"
        self.no_e = f"{self.cp}hasNoEffectOn"
        self.neg_e = f"{self.cp}hasNegativeEffectOn"
        self.effect_triples = self.data[self.data.predicate.isin(
            [self.pos_e, self.no_e, self.neg_e])]

        self.X, self.X_unseen = self.split_triples_unseen()
        self.X = self.X.values

        self.X_train, self.X_valid, self.X_test = self.split_data()
        self.metrics = [
            ("MRR", mrr_score),
            ("Hits@10", lambda ranks: hits_at_n_score(ranks, n=10)),
            ("Hits@3", lambda ranks: hits_at_n_score(ranks, n=3)),
            ("Hits@1", lambda ranks: hits_at_n_score(ranks, n=1)),
        ]

    def describe_data(self):
        """ Describe hypotheses """
        positive_effect = self.data[self.data.predicate == self.pos_e]
        no_effect = self.data[self.data.predicate == self.no_e]
        negative_effect = self.data[self.data.predicate == self.neg_e]
        logger.info(f"# of self.data with positive effect: {positive_effect.shape[0]}")
        logger.info(f"# of self.data with null effect: {no_effect.shape[0]}")
        logger.info(f"# of self.data with negative effect: {negative_effect.shape[0]}")
        logger.info(f"# of self.data depicting effect: {self.effect_triples.shape[0]}")
        logger.info(f"Unique subjects in effect self.data: {self.effect_triples.subject.unique().shape[0]}")
        logger.info(f"Unique predicates in effect self.data: {self.effect_triples.predicate.unique().shape[0]}")
        logger.info(f"Unique objects in effect self.data: {self.effect_triples.object.unique().shape[0]}")
    
    @staticmethod
    def filter_triples(all_df, train_df): 
        """ Filter """
        unseen_triples = all_df.copy()
        for _, row in train_df.iterrows():
            match = all_df[(all_df==row).all(axis=1)]
            unseen_triples = unseen_triples.drop(match.index, axis=0)

        return unseen_triples

    def split_triples_unseen(self):
        """ Split train/unseen """
        s_unique = self.effect_triples.subject.unique()
        p_unique = self.effect_triples.predicate.unique()
        o_unique = self.effect_triples.object.unique()
        all_e_triples = pd.DataFrame(
            list(itertools.product(s_unique, p_unique, o_unique)),
            columns=["subject", "predicate", "object"])
        reg_triples = self.effect_triples.drop_duplicates()
        unseen = self.filter_triples(all_df=all_e_triples, train_df=reg_triples)

        return reg_triples, unseen.values

    def split_data(self):
        """ Split train/eval/test """
        X_train_valid, X_test = train_test_split_no_unseen(self.X, test_size=250)
        X_train, X_valid = train_test_split_no_unseen(X_train_valid, test_size=250)

        print('Train set size: ', X_train.shape)
        print('Test set size: ', X_test.shape)
        print('Valid set size: ', X_valid.shape)
        return X_train, X_valid, X_test
    
    def init_model(self, k: int = 200, eta: int = 15, 
                   scoring_type: str = 'ComplEx', seed: int = 23):
        """ Init Ampligraph model """
        return ScoringBasedEmbeddingModel(
            eta=eta, k=k, scoring_type=scoring_type, seed=seed)
    
    def compile_model(self, model, learning_rate: float = 0.00005, loss: str = 'multiclass_nll',
                      regularizer: str = 'LP', params_regularizer: dict = {'p':1, 'lambda':1e-5},
                      initializer: str = 'glorot_uniform'):
        """ Add optimizer/loss/regularizer/initializer to the model """
        optimizer = Adam(learning_rate=learning_rate)
        loss = get_loss(loss)
        regularizer = get_regularizer(regularizer, params_regularizer)

        model.compile(loss=loss,
                      optimizer=optimizer,
                      entity_relation_regularizer=regularizer,
                      entity_relation_initializer=initializer)
        return model
    
    def fit(self, model, batch_size: int = 200, epochs: int = 500, verbose: bool = True):
        """ Fit model """
        model.fit(self.X_train, batch_size=batch_size, epochs=epochs, verbose=verbose)
        return model
    
    def evaluate_model(self, model, positives_filter, X, corrupt_side: str = 's,o', verbose: bool = True):
        """ Ranking """
        return model.evaluate(X, 
                              use_filter=positives_filter,   # Corruption strategy filter 
                              corrupt_side=corrupt_side,
                              verbose=verbose)
    
    def get_metrics(self, ranks):
        """ Standard metrics """
        res = {}
        for name, metric in self.metrics:
            res[name] = metric(ranks)
        return res
    
    def __call__(self):
        model = self.init_model()
        model = self.compile_model(model)
        model = self.fit(model)

        positives_filter = {'test' : np.concatenate([self.X_train, self.X_valid, self.X_test])}
        ranks = self.evaluate_model(model, positives_filter, self.X_test)
        metrics = self.get_metrics(ranks)
        print("X_test: ", metrics)

        positives_filter['test'] = np.vstack((positives_filter['test'], self.X_unseen))
        ranks = self.evaluate_model(model, positives_filter, self.X_unseen)

        scores = model.predict(self.X_unseen)
        probs = expit(scores)
        df = pd.DataFrame(
            list(zip([' '.join(x) for x in self.X_unseen], 
            ranks, 
            np.squeeze(scores),
            np.squeeze(probs))), 
            columns=['statement', 'rank', 'score', 'prob']) \
                .sort_values("score", ascending=False)
        return df


if __name__ == '__main__':
    TRIPLES = "./triples_107404.csv"
    LP = LinkPredictor(data=TRIPLES)
    DF = LP()
    DF.to_csv("hypotheses.csv")
