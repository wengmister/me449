import numpy as np
import cv2
import matplotlib.pyplot as plt

def demonstrate_dilation():
    # Create a simple map (0 = occupied/unknown, 1 = free)
    free_space = np.zeros((10, 10), dtype=np.uint8)
    free_space[3:7, 3:7] = 1  # Create a free space square in the middle
    
    # Create unknown space (1 = unknown, 0 = known)
    unknown_space = np.ones((10, 10), dtype=np.uint8)
    unknown_space[2:8, 2:8] = 0  # Known space is larger than free space
    
    # Dilate free space
    kernel = np.ones((3, 3), np.uint8)
    free_space_dilated = cv2.dilate(free_space, kernel)
    
    # Find frontiers (dilated free space AND unknown space)
    frontiers = np.logical_and(free_space_dilated, unknown_space)
    
    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    
    axes[0,0].imshow(free_space, cmap='gray')
    axes[0,0].set_title('Original Free Space')
    
    axes[0,1].imshow(free_space_dilated, cmap='gray')
    axes[0,1].set_title('Dilated Free Space')
    
    axes[1,0].imshow(unknown_space, cmap='gray')
    axes[1,0].set_title('Unknown Space')
    
    axes[1,1].imshow(frontiers, cmap='gray')
    axes[1,1].set_title('Detected Frontiers')
    
    for ax in axes.flat:
        ax.grid(True)
    
    plt.tight_layout()
    
    # Return for visualization
    return {
        'free_space': free_space,
        'dilated': free_space_dilated,
        'unknown': unknown_space,
        'frontiers': frontiers
    }

# Create the visualization
result = demonstrate_dilation()

# Print numerical values for clarity
print("\nOriginal Free Space:")
print(result['free_space'])
print("\nDilated Free Space:")
print(result['dilated'])
print("\nFrontiers (where dilated free space meets unknown):")
print(result['frontiers'].astype(int))
plt.show()