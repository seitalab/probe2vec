import argparse

parser = argparse.ArgumentParser()

# Common
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--model', type=str, default="probe2vec01")

# NN
parser.add_argument('--ep', type=int, default=10)
parser.add_argument('--bs', type=int, default=64)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--optim', type=str, default="adam")
parser.add_argument('--device', type=str, default="cpu")
parser.add_argument('--save-every', type=int, default=5)

parser.add_argument('--i_dim', type=int, default=28)
parser.add_argument('--h_dim', type=int, default=64)
parser.add_argument('--negative', type=int, default=5)

args = parser.parse_args()

print(args)
if __name__ == "__main__":
    print("-"*80)
    print(args)
