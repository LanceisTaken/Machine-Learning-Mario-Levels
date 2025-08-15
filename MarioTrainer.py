# Mario Level Generation with Autoencoder - Complete Beginner Guide
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import glob
from sklearn.model_selection import train_test_split

# Step 1: Data Preprocessing
class MarioLevelDataset(Dataset):
    def __init__(self, level_files):
        self.levels = []
        self.char_to_int = self._create_mapping()
        self.int_to_char = {v: k for k, v in self.char_to_int.items()}  # Reverse mapping
        self.original_shapes = []
        
        # Load and process your level files - keep original sizes
        for level_file in level_files:
            level = self._load_level(level_file)
            self.levels.append(level)
            self.original_shapes.append(level.shape)
        
        # Find max dimensions across all levels for padding reference
        max_height = max(shape[0] for shape in self.original_shapes)
        max_width = max(shape[1] for shape in self.original_shapes)
        
        print(f"Loaded {len(self.levels)} levels")
        print(f"Level dimensions range: {min(self.original_shapes)} to {max(self.original_shapes)}")
        print(f"Max dimensions: [{max_height}, {max_width}]")
        print(f"Character mapping: {self.char_to_int}")
    
    def _create_mapping(self):
        # Create mapping from characters to integers
        # Add more characters based on your actual level files
        chars = ['-', '#', 'B', 'p', 'g', 'M', '?', '!', 'E', 'G', 'k', 'r', 'y', 'Y', 'X', 'S', '%', '|', '*']
        char_to_int = {char: i for i, char in enumerate(chars)}
        return char_to_int
    
    def _load_level(self, level_file):
        # Load your actual level files without any resizing
        with open(level_file, 'r') as f:
            lines = f.readlines()
        
        # Convert to integer representation
        level_grid = []
        for line in lines:
            # Remove line numbers and extra characters if present
            clean_line = line.strip()
            if clean_line.startswith(tuple('0123456789')):
                # Remove line numbers (e.g., "1   ----------" becomes "----------")
                parts = clean_line.split(None, 1)
                if len(parts) > 1:
                    clean_line = parts[1]
                else:
                    clean_line = ""
            
            row = []
            for char in clean_line:
                if char in self.char_to_int:
                    row.append(self.char_to_int[char])
                else:
                    row.append(self.char_to_int['-'])  # Default to empty space
            
            if row:  # Only add non-empty rows
                level_grid.append(row)
        
        return np.array(level_grid, dtype=np.int64)  # Use int64 for CrossEntropyLoss
    
    def __len__(self):
        return len(self.levels)
    
    def __getitem__(self, idx):
        level = self.levels[idx]
        
        # For input: normalize to 0-1 range for the encoder
        level_input = torch.FloatTensor(level / len(self.char_to_int)).unsqueeze(0)
        
        # For target: keep as integer indices for CrossEntropyLoss
        level_target = torch.LongTensor(level)
        
        # Return input, target, and original shape information
        return level_input, level_target, level.shape

# Custom collate function to handle different sized levels (updated for classification)
def collate_levels(batch):
    """Custom collate function that pads levels to the same size within each batch"""
    inputs, targets, shapes = zip(*batch)
    
    # Find max dimensions in this batch
    max_height = max(shape[0] for shape in shapes)
    max_width = max(shape[1] for shape in shapes)
    
    # Pad all levels in the batch to the same size
    padded_inputs = []
    padded_targets = []
    
    for inp, tgt in zip(inputs, targets):
        # Current size
        _, curr_h, curr_w = inp.shape
        tgt_h, tgt_w = tgt.shape
        
        # Pad to max size
        pad_h = max_height - curr_h
        pad_w = max_width - curr_w
        
        # Pad inputs with zeros (normalized empty space)
        padded_inp = torch.nn.functional.pad(inp, (0, pad_w, 0, pad_h), value=0)
        
        # Pad targets with 0 (empty space class index)
        padded_tgt = torch.nn.functional.pad(tgt, (0, pad_w, 0, pad_h), value=0)
        
        padded_inputs.append(padded_inp)
        padded_targets.append(padded_tgt)
    
    # Stack into batch tensors
    batch_inputs = torch.stack(padded_inputs)
    batch_targets = torch.stack(padded_targets)
    
    return batch_inputs, batch_targets

# Step 2: Define the Autoencoder Model (Discrete outputs for Mario levels)
class MarioAutoencoder(nn.Module):
    def __init__(self, input_channels=1, num_classes=19):  # 19 different Mario tile types
        super(MarioAutoencoder, self).__init__()
        
        self.num_classes = num_classes
        
        # Encoder - compresses the level into a smaller representation
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d((7, 50)),  # Adaptive pooling to fixed size
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d((4, 25)),  # Further compression
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        
        # Decoder - reconstructs the level from compressed representation
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            
            nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            
            # Output logits for each tile class (not probabilities yet)
            nn.ConvTranspose2d(32, num_classes, kernel_size=3, padding=1),
            # No activation here - we'll use CrossEntropyLoss
        )
    
    def forward(self, x):
        batch_size, channels, height, width = x.shape
        
        # Pass through encoder
        encoded = self.encoder(x)
        
        # Pass through decoder to get logits
        decoded_logits = self.decoder(encoded)
        
        # Resize back to original input size
        final_logits = nn.functional.interpolate(decoded_logits, 
                                                size=(height, width), 
                                                mode='bilinear', 
                                                align_corners=False)
        
        return final_logits
    
    def generate_discrete_output(self, x):
        """Generate discrete tile indices (for visualization/generation)"""
        with torch.no_grad():
            logits = self.forward(x)
            # Get the most probable tile for each position
            discrete_output = torch.argmax(logits, dim=1, keepdim=True).float()
            return discrete_output

# Step 3: Training Function (Updated for classification)
def train_model(model, train_loader, num_epochs=50, learning_rate=0.001):
    # Set device (GPU if available, otherwise CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Define loss function and optimizer for classification
    criterion = nn.CrossEntropyLoss(ignore_index=-1)  # Ignore padding if needed
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    # Training loop
    train_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            # Clear gradients
            optimizer.zero_grad()
            
            # Forward pass - model outputs logits for each class
            output_logits = model(data)
            
            # Reshape for CrossEntropyLoss: (batch, classes, height, width) -> (batch*height*width, classes)
            batch_size, num_classes, height, width = output_logits.shape
            output_logits_reshaped = output_logits.permute(0, 2, 3, 1).contiguous().view(-1, num_classes)
            target_reshaped = target.view(-1)
            
            # Calculate loss
            loss = criterion(output_logits_reshaped, target_reshaped)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Calculate accuracy
            _, predicted = torch.max(output_logits_reshaped, 1)
            correct_predictions += (predicted == target_reshaped).sum().item()
            total_predictions += target_reshaped.size(0)
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        accuracy = 100. * correct_predictions / total_predictions
        train_losses.append(avg_loss)
        
        # Update learning rate
        scheduler.step(avg_loss)
        
        if epoch % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')
    
    return train_losses

# Step 4: Generate New Levels
def generate_level(model, reference_level, device):
    model.eval()
    with torch.no_grad():
        reference_tensor = torch.FloatTensor(reference_level).unsqueeze(0).unsqueeze(0).to(device)
        generated = model(reference_tensor)
        return generated.cpu().squeeze().numpy()

# Step 5: Visualization (Updated for discrete outputs)
def visualize_levels(original, reconstructed, dataset=None, generated=None):
    """Visualize original and reconstructed levels with proper character mapping"""
    
    # Convert reconstructed logits/indices back to character display
    if dataset is not None:
        int_to_char = dataset.int_to_char
        
        # Create text representation for better visualization
        def array_to_text(arr):
            if len(arr.shape) == 2:  # 2D array
                text_lines = []
                for row in arr:
                    line = ''.join([int_to_char.get(int(val), '-') for val in row])
                    text_lines.append(line)
                return '\n'.join(text_lines)
            return str(arr)
        
        print("Original Level:")
        print(array_to_text(original))
        print("\nReconstructed Level:")
        print(array_to_text(reconstructed))
        
        if generated is not None:
            print("\nGenerated Level:")
            print(array_to_text(generated))
    
    # Visual plot
    fig, axes = plt.subplots(1, 3 if generated is not None else 2, figsize=(20, 6))
    
    axes[0].imshow(original, cmap='tab20', aspect='auto')
    axes[0].set_title('Original Level')
    axes[0].axis('off')
    
    axes[1].imshow(reconstructed, cmap='tab20', aspect='auto')
    axes[1].set_title('Reconstructed Level')
    axes[1].axis('off')
    
    if generated is not None:
        axes[2].imshow(generated, cmap='tab20', aspect='auto')
        axes[2].set_title('Generated Level')
        axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()

# Step 6: Main Training Script
def main():
    # 1. Prepare your data
    # Replace this with paths to your actual level files
    level_files = glob.glob(r"C:\Users\User\Documents\GitHub\Machine-Learning---Mario-Levels\Levels\*.txt")  # Your level files
    
    # You can also automatically find all .txt files in a directory:
    # import glob
    # level_files = glob.glob("path/to/your/levels/*.txt")
    
    # Create dataset and split into train/validation
    try:
        # No resizing - keep original level dimensions
        dataset = MarioLevelDataset(level_files)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        # Create data loaders with custom collate function
        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_levels)
        val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=collate_levels)
        
        print(f"Dataset loaded: {len(dataset)} levels")
        print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
        
    except FileNotFoundError as e:
        print(f"Level files not found: {e}")
        print("Please update the file paths in level_files list.")
        return
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # 2. Create and train the model  
    model = MarioAutoencoder()  # Now adaptive to any input size
    print("Model created:", model)
    
    # 3. Train the model
    print("Starting training...")
    train_losses = train_model(model, train_loader, num_epochs=100, learning_rate=0.001)
    
    # 4. Save the trained model
    torch.save(model.state_dict(), 'mario_autoencoder.pth')
    print("Model saved as 'mario_autoencoder.pth'")
    
    # 5. Test the model (generate a level) - Updated for discrete output
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    # Get a sample from validation set
    sample_data, sample_target = next(iter(val_loader))
    
    with torch.no_grad():
        # Get discrete reconstruction
        reconstructed_discrete = model.generate_discrete_output(sample_data.to(device))
    
    # Convert to numpy for visualization
    original = sample_target[0].cpu().numpy()
    reconstructed = reconstructed_discrete[0].cpu().squeeze().numpy()
    
    # Visualize results with character mapping
    visualize_levels(original, reconstructed, dataset)
    
    # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses)
    plt.title('Training Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('CrossEntropy Loss')
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
            # Denormalize from 0-1 range back to character indices
            char_idx = int(round(val * len(int_to_char)))
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