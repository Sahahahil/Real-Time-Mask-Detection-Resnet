import matplotlib.pyplot as plt

def plot_loss(losses):
    plt.figure(figsize=(8, 5))
    plt.plot(losses, marker='o')
    plt.title('Training Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid()
    plt.tight_layout()
    plt.savefig('loss_curve.png')
    plt.show()
    print("Loss curve saved as loss_curve.png")