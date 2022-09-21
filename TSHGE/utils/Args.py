import  argparse

args = argparse.ArgumentParser()
args.add_argument('--dataset', type=str, default='YAGO')
args.add_argument('--time-interval', type=int, default=1)#1
args.add_argument('--lr-gat', type=float, default=0.001)#0.001
args.add_argument('--lr-conv', type=float, default=0.001)#0.001
args.add_argument('--n-epochs-gat', type=int, default=40)#40 for YAGO, 60 for ICEWS14s, 50 for WIKI
args.add_argument('--n-epochs-conv', type=int, default=10)#50
args.add_argument('--embedding-dim', type=int, default=200)
args.add_argument('--embedding-dim1', type=int, default=10)
args.add_argument('--hidden-dim', type=int, default=12800)#12800
args.add_argument('--dropout-ta', type=float, default=0.2)
args.add_argument('--input-drop', type=float, default=0.2)
args.add_argument('--hidden-drop', type=float, default=0.4)
args.add_argument('--feat-drop', type=float, default=0.3)
args.add_argument('--batch-size-conv', type=int, default=2048)#2048 for YAGO, 2560 for WIKI, 1280 for ICEWS14
args.add_argument("-alpha", "--alpha", type=float, default=0.2, help="LeakyRelu alphs for SpGAT layer")
args.add_argument("-margin", "--margin", type=float, default=5, help="Margin used in hinge loss")
args.add_argument('--pred', type=str, default='sub')
args.add_argument("--reg-para", type=float, default=0.01)
args.add_argument('--valid-epoch', type=int, default= 15)
args.add_argument('--count', type=int, default=5)#3
args.add_argument("--train-history-length", type=int, default=5)#5 for YAGO and WIKI, 10 for ICEWS14s
args.add_argument("--test-history-length", type=int, default=5)#5 for YAGO and WIKI, 10 for ICEWS14s
args.add_argument("--multi-step", action='store_true', default=True)
args.add_argument("--topk", type=int, default=10,
                    help="choose top k entities as results when do multi-steps without ground truth")
args.add_argument('--seed', type=int, default=42)#42

args, unknown = args.parse_known_args()
print(args)
