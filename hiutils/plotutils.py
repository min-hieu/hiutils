import math
import matplotlib.pyplot as plt

def plot_lists(*imgLL, orientation="vertical"):
    N = len(imgLL[0])   # len of every image lists
    L = len(imgLL)      # number of image lists
    assert(len(imgL) == N for imgL in imgLL)

    if orientation == "horizontal":
        fig = plt.figure(figsize=(3*N+1, 3*L+1))
    elif orientation == "vertical":
        fig = plt.figure(figsize=(3*L+1, 3*N+1))
    else:
        raise Exception("invalid orientation")

    fig.subplots_adjust(wspace=0, hspace=0)
    for i in range(L*N):
        if orientation == "horizontal":
            ax = fig.add_subplot(N, L, i + 1).imshow(imgLL[i%L][i])
        elif orientation == "vertical":
            ax = fig.add_subplot(L, N, i + 1).imshow(imgLL[i%L][i])
        else:
            raise Exception("invalid orientation")
        plt.axis("off")

    return fig

def plot_square(img_list):
    L = len(img_list)
    H = int(L ** 0.5)
    W = int(math.ceil(L / H))

    fig = plt.figure(figsize=(W*3+1, H*3+1))
    fig.subplots_adjust(wspace=0, hspace=0)
    for i in range(L):
        ax = fig.add_subplot(W, H, i + 1).imshow(img_list[i])
        plt.axis("off")

    return fig
