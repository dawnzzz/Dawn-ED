import argparse


class Args:
  @staticmethod
  def parse():
    parser = argparse.ArgumentParser()
    return parser

  @staticmethod
  def initialize(parser):
    # SELECT MODEL
    parser.add_argument('--model', default='DMBERT', help='the model you selected to train/test')

    # args for path
    parser.add_argument('--output_dir', default='./checkpoints/',
                        help='the output dir for model checkpoints')

    parser.add_argument('--bert_dir', default='../pretrain/chinese-roberta-wwm-ext',
                        help='bert dir for uer')
    parser.add_argument('--data_dir', default='./data/',
                        help='data dir for uer')
    parser.add_argument('--log_dir', default='./logs/',
                        help='log dir for uer')

    # other args
    parser.add_argument('--num_tags', default=66, type=int,
                        help='number of tags')

    parser.add_argument('--gpu_ids', type=str, default='0,1',
                        help='gpu ids to use, -1 for cpu, "0/1" for gpu')

    parser.add_argument('--max_seq_len', default=256, type=int)

    parser.add_argument('--eval_batch_size', default=12, type=int)

    parser.add_argument('--swa_start', default=3, type=int,
                        help='the epoch when swa start')

    # train args
    parser.add_argument('--train_epochs', default=10, type=int,
                        help='Max training epoch')

    parser.add_argument('--dropout_prob', default=0.1, type=float,
                        help='drop out probability')

    parser.add_argument('--lr', default=2e-5, type=float,
                        help='learning rate for the bert module')

    parser.add_argument('--other_lr', default=2e-4, type=float,
                        help='learning rate for the module except bert')

    parser.add_argument('--max_grad_norm', default=1.0, type=float,
                        help='max grad clip')

    parser.add_argument('--warmup_proportion', default=0.1, type=float)

    parser.add_argument('--weight_decay', default=0.01, type=float)

    parser.add_argument('--adam_epsilon', default=1e-8, type=float)

    parser.add_argument('--train_batch_size', default=20, type=int)

    # args for DMCNN
    parser.add_argument('--pf_dim', default=5, type=int)

    parser.add_argument('--llf_num', default=5, type=int)

    parser.add_argument('--hidden_size', default=200, type=int)

    parser.add_argument('--kernel_size', default=3, type=int)

    return parser

  def get_parser(self):
    parser = self.parse()
    parser = self.initialize(parser)
    return parser.parse_args()
