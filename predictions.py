import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

def pred_and_plot_image(model_path, image_path, class_names, image_size=(224, 224), transform=None, device="cuda"):
    # Load the model from the provided path
    model_state = torch.load(model_path, map_location=device)
    model = model_state['model']  # Extract the model from the state dictionary
    model.eval()  # Set model to evaluation mode

    # Define transformation if not provided
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    # Load and transform the image
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)

    # Move image to the target device
    image = image.to(device)

    # Move model to the target device
    model = model.to(device)

    # Perform inference
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        predicted_class = class_names[predicted.item()]

    # Plot the image with predicted class
    plt.imshow(image.squeeze().permute(1, 2, 0).cpu())
    plt.axis('off')
    plt.title(f'Predicted class: {predicted_class}')
    plt.show()

