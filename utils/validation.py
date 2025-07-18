import matplotlib.pyplot as plt
import torch

def decode_label(label_tensor, full_dataset):
    idx_to_char = {i: c for i, c in enumerate(full_dataset.char_set)}

    return ''.join([idx_to_char[int(i)] for i in label_tensor])

def predict(model, images):
    model.eval()

    with torch.no_grad():
        outputs = model(images)
        predictions = torch.argmax(outputs, dim=2)

    return predictions


def visualize_predictions(full_dataset, model, loader, num_examples=5):
    images, labels = next(iter(loader))
    images = images[:num_examples]
    labels = labels[:num_examples]
    preds = predict(model, images)

    for i in range(num_examples):
        img = images[i].permute(1, 2, 0)  # (C, H, W) -> (H, W, C)
        true_label = decode_label(labels[i], full_dataset)
        pred_label = decode_label(preds[i], full_dataset)

        plt.imshow(img.squeeze(), cmap='gray')
        plt.title(f"True: {true_label} | CNN Predict: {pred_label}")
        plt.axis('off')
        plt.show()
