import argparse
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch_size', type=int, default=1280, metavar='N',
                    help='input batch size for training (default: 64)')  # 往parser中填充参数

parser.add_argument('-b', type=int, default=[7,8,9], metavar='n', nargs=3)

args = parser.parse_args()
print(args)
print(args.batch_size)#中间的小横线被解析成下划线#'--batch_size'或者'--batch-size'都可以
print(args.b)#可选参数“- -”和一个“-”都有的话，默认是两个的属性，一个的没有

# parser.add_argument("echo") #　默认必选
# args = parser.parse_args()
# print (args.echo)