# Hopfield Network
Hopfield network (Amari-Hopfield network) implemented with Python. Two update rules are implemented: **Asynchronous** & **Synchronous**.

## Requirement
- Python >= 3.5
- numpy
- matplotlib
- skimage
- tqdm
- keras (to load MNIST dataset)

## Usage
Run `train.py` or `train_mnist.py`.

## Demo

### train.py
The following is the result of using **Synchronous** update.
```
Start to data preprocessing...
Start to train weights...
100%|██████████| 4/4 [00:06<00:00,  1.67s/it]
Start to predict...
100%|██████████| 4/4 [00:02<00:00,  1.80it/s]
Show prediction results...
```


![](https://raw.githubusercontent.com/Tafadzwa-Chigumira/hopfield-network-lab5/main/results/result_6.png)

1. **Train Data:** The original stored patterns.

2. **Input Data:** The same patterns with 20% corruption (noise).

3. **Output Data:** The recalled patterns after passing through the Hopfield network.

#### **Observations**

· The **Input Data** (middle column) contains significant noise, yet the overall structure of the patterns remains somewhat recognizable.

· The **Output Data** (right column) shows that the Hopfield network successfully denoises and retrieves the original patterns, demonstrating robust memory recall at **20% corruption**.

· While some minor distortions remain, the network effectively reconstructs the key features of each image, indicating that at this noise level, the system retains high recall accuracy.

![](https://raw.githubusercontent.com/Tafadzwa-Chigumira/hopfield-network-lab5/main/results/result_5.png)

1. **Train Data (Left Column)**: Original stored patterns that the network was trained on.

2. **Input Data (Middle Column)**: Noisy versions of the stored patterns after applying corruption (at 0.3 in this case).

3. **Output Data (Right Column)**: Recovered patterns after the Hopfield network attempts to retrieve them.

### **Observations at 30% Noise:**

· The **Input Data** column shows patterns that have been significantly distorted by noise (random pixel flips).

· The **Output Data** column shows the network's attempt to recover the original patterns.

· The network successfully retrieves most of the original structures, but some fine details may be lost or misrepresented depending on the noise level.

![](https://raw.githubusercontent.com/Tafadzwa-Chigumira/hopfield-network-lab5/main/results/result_4.png)

1. **Train Data (Left Column)**: These are the original stored patterns that the Hopfield network has learned.

2. **Input Data (Middle Column)**: The input data has undergone corruption, with a significant amount of noise (40%). The images appear highly distorted with a high density of pixel inversions.

3. **Output Data (Right Column)**: Despite the corruption in the input, the Hopfield network has been able to reconstruct the original patterns to a recognizable degree.

### **Observations at 40% Noise:**

· **Pattern Recovery**: The network is still able to reconstruct the original images, but there might be minor distortions.

· **Robustness**: Even with nearly half the pixels being flipped, the network successfully retrieves stored images, demonstrating its associative memory capabilities.

· **Threshold Effect**: While the network performs well at 40%, further increasing noise (e.g., 50%) might cause the system to fail in recovering the patterns.

![](https://raw.githubusercontent.com/Tafadzwa-Chigumira/hopfield-network-lab5/main/results/result.png)

### Image 1 (Corruption Level: 0.2)

- **Input Data**: Shows mild corruption with approximately 20% of pixels zeroed out
- **Output Data**: Very successful recovery with almost perfect reconstruction
- The fine details in all four test images are well preserved
- Even subtle features in the silhouette outlines remain intact

![](https://raw.githubusercontent.com/Tafadzwa-Chigumira/hopfield-network-lab5/main/results/result_1.png)

### Image 2 (Corruption Level: 0.3)

- **Input Data**: Moderate corruption with noticeable degradation
- **Output Data**: Still strong recovery with minimal loss of detail
- Some very fine details in the top portrait show slight imperfections
- The overall structure of all patterns remains highly recognizable

![](https://raw.githubusercontent.com/Tafadzwa-Chigumira/hopfield-network-lab5/main/results/result_2.png)

### Image 3 (Corruption Level: 0.4)

- **Input Data**: Significant corruption with substantial noise
- **Output Data**: Good recovery but with more visible artifacts
- The third image (person in motion) shows some minor distortion in the recovered version
- Edge definition is slightly reduced compared to lower corruption levels

![](https://raw.githubusercontent.com/Tafadzwa-Chigumira/hopfield-network-lab5/main/results/result_3.png)

### Image 4 (Corruption Level: 0.5)

- **Input Data**: Severe corruption with half of pixels potentially zeroed
- **Output Data**: Surprisingly effective recovery despite extreme corruption
- Some loss of fine detail, particularly in the bottom image
- Minor artifacts appear in complex regions but overall structure is maintained

## Comparative Observations

1. As corruption level increases from 0.2 to 0.5, recovery quality gradually degrades
2. The system shows remarkable robustness even at 50% corruption
3. Performance varies slightly by image complexity - simpler shapes (like the second image) recover better at high corruption
4. Zero-type corruption (replacing values with zero rather than inverting) appears to be less destructive to pattern recognition than noise-type corruption[README.md](README.md)

## Reference
- Amari, "Neural theory of association and concept-formation", SI. Biol. Cybernetics (1977) 26: 175. https://doi.org/10.1007/BF00365229
- J. J. Hopfield, "Neural networks and physical systems with emergent collective computational abilities", Proceedings of the National Academy of Sciences of the USA, vol. 79 no. 8 pp. 2554–2558, April 1982.