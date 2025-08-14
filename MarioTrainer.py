# Mario Level Generation with Autoencoder - Complete Beginner Guide
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Step 1: Data Preprocessing
class MarioLevelDataset(Dataset):
    def __init__(self, level_files):
        self.levels = []
        self.char_to_int = self._create_mapping()
        
        # Load and process your level files
        for level_file in level_files:
            level = self._load_level(level_file)
            self.levels.append(level)
    
    def _create_mapping(self):
        # Create mapping from characters to integers
        # Add more characters based on your actual level files
        chars = ['-', '#', 'B', 'p', 'g', 'M', '?', '!', 'E', 'G', 'k', 'r', 'y', 'Y', 'X', 'S', '%', '|', '*']
        char_to_int = {char: i for i, char in enumerate(chars)}
        char_to_int['PAD'] = len(chars)  # For padding shorter levels
        return char_to_int
    
    def _load_level(self, level_file):
        # This is where you'd load your actual level files
        # For now, I'll show you how to process the format you showed
        with open(level_file, 'r') as f:
            lines = f.readlines()
        
        # Convert to integer representation
        level_grid = []
        for line in lines:
            row = []
            for char in line.strip():
                if char in self.char_to_int:
                    row.append(self.char_to_int[char])
                else:
                    row.append(self.char_to_int['-'])  # Default to empty space
            level_grid.append(row)
        
        return np.array(level_grid, dtype=np.float32)
    
    def __len__(self):
        return len(self.levels)
    
    def __getitem__(self, idx):
        level = self.levels[idx]
        # Convert to tensor and add batch dimension for CNN
        level_tensor = torch.FloatTensor(level).unsqueeze(0)  # Add channel dimension
        return level_tensor, level_tensor  # Input and target are the same for autoencoder

# Step 2: Define the Autoencoder Model
class MarioAutoencoder(nn.Module):
    def __init__(self, input_channels=1):
        super(MarioAutoencoder, self).__init__()
        
        # Encoder - compresses the level into a smaller representation
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Reduce size by half
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Reduce size by half again
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        
        # Decoder - reconstructs the level from compressed representation
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),  # Double the size
            
            nn.ConvTranspose2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),  # Double the size again
            
            nn.ConvTranspose2d(16, input_channels, kernel_size=3, padding=1),
            nn.Sigmoid()  # Output values between 0 and 1
        )
    
    def forward(self, x):
        # Pass through encoder then decoder
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Step 3: Training Function
def train_model(model, train_loader, num_epochs=50, learning_rate=0.001):
    # Set device (GPU if available, otherwise CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()  # Mean Squared Error for reconstruction
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    train_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            # Clear gradients
            optimizer.zero_grad()
            
            # Forward pass
            output = model(data)
            loss = criterion(output, target)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        if epoch % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')
    
    return train_losses

# Step 4: Generate New Levels
def generate_level(model, reference_level, device):
    model.eval()
    with torch.no_grad():
        reference_tensor = torch.FloatTensor(reference_level).unsqueeze(0).unsqueeze(0).to(device)
        generated = model(reference_tensor)
        return generated.cpu().squeeze().numpy()

# Step 5: Visualization
def visualize_levels(original, reconstructed, generated=None):
    fig, axes = plt.subplots(1, 3 if generated is not None else 2, figsize=(15, 5))
    
    axes[0].imshow(original, cmap='viridis')
    axes[0].set_title('Original Level')
    axes[0].axis('off')
    
    axes[1].imshow(reconstructed, cmap='viridis')
    axes[1].set_title('Reconstructed Level')
    axes[1].axis('off')
    
    if generated is not None:
        axes[2].imshow(generated, cmap='viridis')
        axes[2].set_title('Generated Level')
        axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()

# Step 6: Main Training Script
def main():
    # 1. Prepare your data
    # Replace this with paths to your actual level files
    level_files = ['level1.txt', 'level2.txt', 'level3.txt']  # Your level files
    
    # Create dataset and split into train/validation
    try:
        dataset = MarioLevelDataset(level_files)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
        
        print(f"Dataset loaded: {len(dataset)} levels")
        print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
        
    except FileNotFoundError:
        print("Level files not found. Please update the file paths in level_files list.")
        return
    
    # 2. Create and train the model
    model = MarioAutoencoder()
    print("Model created:", model)
    
    # 3. Train the model
    print("Starting training...")
    train_losses = train_model(model, train_loader, num_epochs=100, learning_rate=0.001)
    
    # 4. Save the trained model
    torch.save(model.state_dict(), 'mario_autoencoder.pth')
    print("Model saved as 'mario_autoencoder.pth'")
    
    # 5. Test the model (generate a level)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    # Get a sample from validation set
    sample_data, _ = next(iter(val_loader))
    with torch.no_grad():
        reconstructed = model(sample_data.to(device))
    
    # Visualize results
    original = sample_data[0].squeeze().numpy()
    recon = reconstructed[0].cpu().squeeze().numpy()
    
    visualize_levels(original, recon)
    
    # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses)
    plt.title('Training Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

if __name__ == "__main__":
    main()

# Additional Helper Functions

def load_pretrained_model(model_path):
    """Load a previously trained model"""
    model = MarioAutoencoder()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def convert_level_to_text(level_array, int_to_char=None):
    """Convert integer array back to readable text format"""
    if int_to_char is None:
        # Default mapping (reverse of what we used earlier)
        chars = ['-', '#', 'B', 'p', 'g', 'M', '?', '!', 'E', 'G', 'k', 'r', 'y', 'Y', 'X', 'S', '%', '|', '*']
        int_to_char = {i: char for i, char in enumerate(chars)}
    
    text_level = []
    for row in level_array:
        text_row = ""
        for val in row:
            char_idx = int(round(val))
            char_idx = max(0, min(char_idx, len(int_to_char)-1))  # Clamp to valid range
            text_row += int_to_char.get(char_idx, '-')
        text_level.append(text_row)
    
    return text_level

# Usage example for generating new levels:
"""
# After training, load your model and generate new levels:

model = load_pretrained_model('mario_autoencoder.pth')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load a reference level
reference_level = np.array([...])  # Your level data

# Generate new level
new_level = generate_level(model, reference_level, device)

# Convert back to text format
text_level = convert_level_to_text(new_level)
for row in text_level:
    print(row)
"""