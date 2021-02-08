import matplotlib.pyplot as plt


def visualize(im, imAgmented, operation):
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.title('Imagen original')
    plt.imshow(im)
    plt.subplot(1, 2, 2)
    plt.title(operation)
    plt.imshow(imAgmented)
