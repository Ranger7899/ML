import torch
from PIL import Image
from torchvision import transforms

from CNN_Digit_Trainer import SimpleCNN

def test_model_with_image(model, image_path, device):
    """
    Test the trained model with an input image (.png file).

    Args:
        model (nn.Module): Trained PyTorch model.
        image_path (str): Path to the input .png image file.
        device (torch.device): Device (CPU or CUDA) for inference.

    Returns:
        int: Predicted class label.
    """
    # Define preprocessing steps
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Ensure grayscale
        transforms.Resize((28, 28)),                 # Resize to 28x28
        transforms.ToTensor(),                       # Convert to tensor
        transforms.Normalize((0.5,), (0.5,))         # Normalize to [-1, 1]
    ])

    # Load and preprocess the image
    image = Image.open(image_path)
    input_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension

    # Perform inference
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.exp(output)  # Convert log probabilities to probabilities
        predicted_class = torch.argmax(probabilities, dim=1).item()

    return predicted_class

# Example usage
if __name__ == "__main__":
    # Load the trained model
    model = SimpleCNN()  # Replace with your model class name
    model.load_state_dict(torch.load('CNN/simple_cnn.pth')['model_state_dict'])
    model = model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    # Test with an input image
    image_path = "C:/Users/ymasa/Desktop/Simple-Neural-Network-for-Handwritten-Digit-Recognition/Handwritten_Images/handwritten_0.png"  # Replace with your .png file path
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predicted_class = test_model_with_image(model, image_path, device)

    print(f"The model predicts the image as class: {predicted_class}")
