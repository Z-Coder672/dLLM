import re
import matplotlib.pyplot as plt

with open('training.log', 'r') as f:
    lines = f.readlines()

train_steps, train_losses = [], []
val_steps, val_losses = [], []

for line in lines:
    # Match training loss: 'Step   1760 | Loss: 4.8631'
    train_match = re.search(r'Step\s+(\d+)\s+\|\s+Loss:\s+([\d.]+)', line)
    if train_match:
        train_steps.append(int(train_match.group(1)))
        train_losses.append(float(train_match.group(2)))
    
    # Match validation loss: '  Val Loss: 4.4150'
    val_match = re.search(r'Val Loss:\s+([\d.]+)', line)
    if val_match and train_steps:
        val_steps.append(train_steps[-1])  # Use last training step
        val_losses.append(float(val_match.group(1)))

plt.figure(figsize=(12, 6))
plt.plot(train_steps, train_losses, alpha=0.3, label='Training Loss', linewidth=0.5)
plt.plot(val_steps, val_losses, 'o-', label='Validation Loss', linewidth=2, markersize=4)
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.title('Training Progress')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('loss_curve.png', dpi=150)
print('Saved to loss_curve.png')
plt.show()