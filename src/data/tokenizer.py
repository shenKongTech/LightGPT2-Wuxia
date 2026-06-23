import os
import sentencepiece as spm
from typing import List, Optional

class BpeTokenizer:
    def __init__(self,model_prefix,vocab_size,train_corpus = None):
        self.model_prefix = model_prefix
        self.vocab_size = vocab_size
        self.sp = None
        self.model_path =f'{model_prefix}.model'

        if os.path.exists(self.model_path):
            self._load_model()
        elif train_corpus is not None:
            self._train_model(train_corpus)
            self._load_model()
        else:
            raise FileNotFoundError('未找到BPE分词器模型文件，请先运行训练模式生成.')

    def _train_model(self,text):
        print('正在训练BPE分词器...')
        temp_corpus = 'temp_wuxia_corpus.txt'
        with open(temp_corpus,'w',encoding='utf-8') as f :
            f.write(text)

        spm.SentencePieceTrainer.train(
            input = temp_corpus,
            model_prefix = self.model_prefix,
            model_type = 'bpe',
            character_coverage = 0.9995,
            pad_id = 0,
            unk_id = 1,
            bos_id = 2,
            eos_id = 3,
            control_symbols = ['<PAD>','<UNK>','<BOS>','<EOS>'],
            input_sentence_size = 1000000,
            shuffle_input_sentence = True,
            num_threads = 20
        )
        os.remove(temp_corpus)
        print(f'BPE分词器一已训练完成，总词数：{self.vocab_size}')

    def _load_model(self):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(self.model_path)

    def encode(self,text,add_special_tokens = True):
        ids = self.sp.encode(text,out_type=int)
        if add_special_tokens:
            return [self.sp.bos_id()] + ids + [self.sp.eos_id()]
        return ids

    def decode(self,ids):
        special_ids = {self.sp.pad_id(),self.sp.unk_id(),self.sp.bos_id(),self.sp.eos_id()}
        clean_ids = [i for i  in ids if i not in special_ids]
        return self.sp.decode(clean_ids)

    @property
    def vocab_len(self):
        return self.sp.get_piece_size()
