# Test the model on a few images
import numpy as np
def imshow(img, title):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(title)
    plt.show()

# Assuming 'train_loader' is your DataLoader
dataiter = iter(val_loader)

# Correct way to iterate over DataLoader
try:
    images, labels = next(dataiter)
    # Process inputs and labels
except StopIteration:
    # Handle end of iterator
    pass



outputs = model(images)
_, predicted = torch.max(outputs, 1)

# Display the first 4 images with their predictions
for i in range(15,17):
    imshow(images[i].cpu(), f'Predicted: {dataset.classes[predicted[i]]}')
