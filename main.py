import torch
from trainer import Trainer
from data import Data
from model import GPTLanguageModel
 

class static_variables() :
    BATCH_SIZE = 16 
    BLOCK_SIZE = 8 
    MAX_ITERS = 100
    EVAL_INTERVAL = 10
    LEARNING_RATE = 0.001
    EVAL_ITERS = 200
    RANDOM_SEED = 1337
    WORD_COUNT = 100
    n_embd =  32
    n_head =  4
    n_layer =  3
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dropout = 0.2
    DATA_FILE = "input.txt"


def main():
    torch.manual_seed(static_variables.RANDOM_SEED)
    data = Data(static_variables.DATA_FILE)
    model = GPTLanguageModel(data.vocab_size,static_variables)
    trainer = Trainer(data.get_data(), model,static_variables)
    context =trainer.train(static_variables.MAX_ITERS)
    generated = model.generate(context, static_variables.WORD_COUNT)[0].tolist()
    print(data.decode(generated))

if __name__ == "__main__":
    main()
