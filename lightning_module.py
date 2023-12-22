#%% libraries
import torch
from torch.nn.functional import nll_loss
from lightning import LightningModule

#%% body
class LightningRE(LightningModule):
    def __init__(self, lm, ner, clusterer):
        super().__init__()
        self.lm = lm
        self.ner = ner
        self.clusterer = clusterer
        self.rc = rc

    def lm_step(self, example):
        tokens = example.subword_tokens
        token_embeddings = self.lm(tokens)

        return token_embeddings

    def universal_ner_step(self, example, token_embeddings):
        logits = self.ner(example.candidate_spans, token_embeddings)
        return logits

    def ner_training_step(self, example, token_embeddings):
        logits = self.universal_ner_step(example, token_embeddings)
        
        loss = nll_loss(logits, example.train_candidate_span_labels)
        
        return loss

    def ner_predict_step(self, example, token_embeddings):
        logits = self.universal_ner_step(example, token_embeddings)
        self.ner.predict(example.spans, logits)
        

    def universal_cluster_training_step(self, example):
        logits = self.clusterer(example.)
    
    def re_training_step(self, example):
        

    def training_step(self, example, example_index):
        
        

    def validation_step(self):

    def test_step(self):

    def configure_optimizers(self):

    