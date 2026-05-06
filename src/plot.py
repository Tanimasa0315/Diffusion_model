import matplotlib.pyplot as plt

class Plotter:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.label_dict = self.label_to_class_name(dataset_name)

    def label_to_class_name(self, dataset_name):
        if dataset_name == "MNIST":
            label_dict = {i: str(i) for i in range(10)}
        elif dataset_name == "FashionMNIST":
            label_dict = {0: 'T-shirt/top', 1: 'Trouser', 2: 'Pullover', 3: 'Dress', 4: 'Coat', 5: 'Sandal', 6: 'Shirt', 7: 'Sneaker', 8: 'Bag', 9: 'Ankle boot'}
        elif dataset_name == "CIFAR10":
            label_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'}
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        return label_dict

    def show_images_cond(self, images, labels, rows=2, cols=10):
        fig, ax = plt.subplots(rows, cols, figsize=(10, 2.5), sharey=True)
        i = 0
        for r in range(rows):
            for c in range(cols):
                ax[r, c].imshow(images[i], cmap='gray')
                ax[r, c].set_title(f"{labels[i]}: {self.label_dict.get(int(labels[i]), 'Unknown')}", ha="center", fontsize=10)
                ax[r, c].axis('off')
                i += 1
        plt.tight_layout()
        plt.show()
        plt.close()

    def show_images(self, images, rows=2, cols=10):
        fig = plt.figure(figsize=(cols, rows))
        i = 0
        for r in range(rows):
            for c in range(cols):
                fig.add_subplot(rows, cols, i + 1)
                plt.imshow(images[i], cmap='gray')
                plt.axis('off')
                i += 1
        plt.show()
        plt.close()

    def plot_loss(self, train_losses, save_path=None):
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Train Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Over Epochs')
        plt.legend()
        plt.grid()
        if save_path is not None:
            plt.savefig(save_path)
        plt.show()
        plt.close()