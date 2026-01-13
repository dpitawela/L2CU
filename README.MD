# L2CU: Learning To Complement Unseen Users
<a href="https://ieeexplore.ieee.org/document/11314492">
  <img src="./assets/ieee_logo.png" alt="IEEE Logo" width="70"/>
</a>
<a href="https://arxiv.org/abs/2601.06119">
  <img src="./assets/arxiv_logo.jpg" alt="Arxiv Logo" width="50"/>
</a>

## Overview

- Below are basic code snippets that illustrate the core components of the framework, including:
  - Model definition  
  - Loss function  
  - Data loader  
  - Training loop
- A complete code example based on the CIFAR10N experiment is provided in the `codes/` directory that goes through Python notebooks 1 to 5.
- `code/matrics.py` contains the implementations of the evaluation criteria.

### 🧠 Basic Model Architecture
The proposed architecture consists of three main components:

1. **Base Model**  
   Extracts visual features from the input data.

2. **Human Label Encoder**  
   Encodes annotator-specific labeling behavior.

3. **Decision Model**  
   Combines the visual features and encoded human labels to produce the final prediction.

An example of this architecture is shown below for the CIFAR-10N dataset, which contains 10 classes.

> **Note:** Models are trained separately for each annotator profile.

```python
class  AdaptedAI(nn.Module):
	def  __init__(self):
		super(AdaptedAI, self).__init__()
		n_classes = 10 # CIFAR-10N contains 10 classes

		# --------------------------------------------------
		# Base Model
		# --------------------------------------------------
		# Any classification backbone can be used here (e.g., ResNet, Transformer).
		# The model should output logits or feature representations of size [batch_size, n_classes].
		self.base_model = 
		
		# --------------------------------------------------
		# Human Label Encoder
		# --------------------------------------------------
		# Encodes annotator-provided labels into a latent representation
		# that captures annotator-specific labeling behavior.
		self.n_l_encoder = nn.Sequential(
			nn.Linear(n_classes, 32),
			nn.ReLU(),
			nn.Linear(32, n_classes)
		)  
		
		# --------------------------------------------------
		# Decision Model
		# --------------------------------------------------
		# Combines image-based features and encoded human label features
		# to produce the final classification prediction.
		self.decision_ai = nn.Sequential(
			nn.Linear(2*n_classes, 64),
			nn.ReLU(),
			nn.Linear(64, 32),
			nn.ReLU(),
			nn.Linear(32, n_classes),
		)
	
	def  forward(self, imgs, n_l):
		# Extract visual features from input images
		img_features = self.base_model(imgs)

		# Encode human (annotator) labels
		n_l_features = self.n_l_encoder(n_l)

		# Concatenate image features and encoded label features
		out = torch.cat((img_features, n_l_features), dim=1)

		# Compute final prediction
		out = self.decision_ai(out)
		return img_features, n_l, out
```

### 📑 Basic Data Loader
> **Note:** Data are loaded per annotator profile

```python
class  CIFAR10(VisionDataset):
	def  __init__(self):
		# Initialize dataset paths, load images, consensus labels,
		# and annotator profile specific noisy labels
		pass
	
	def  __len__(self) -> int:
		return  len(self.noisyLabels)
	
	def  __getitem__(self, index: int) -> Tuple[Any, Any]:
		# Noisy labels are sampled from the annotator profile,
		# while images and consensus labels are aligned accordingly.
		img, consensus, n_l = self.data[index % len(self.data)]
		consensus = self.consensus[index % len(self.data)]
		n_l = self.noisyLabels[index]
		
		# Apply image transforms (if any)

		# Return: image, its consensus label, one of its noisy labels from 
		# the annotator profile
		return img, consensus, n_l
```

### 🧮 Loss Function
```python
class  CorrectionLoss(nn.Module):
	def  __init__(self, loss1, C=0, N_human=None, N_base=None):
		super().__init__()
		self.loss1 = loss1 # a standard classification loss (e.g., cross entropy)
		self.C = C # weighting factor for the noise correction term (optional)
		self.N_h = N_human # noise transition matrix modeling human annotator noise (optional)
		self.N_b = N_base # noise transition matrix modeling base model prediction noise (optional)

	def  noiseCorrection(self, prediction, n_input):
		softmax_pred = F.softmax(prediction, dim=1)
		correction = torch.tensor([]).to(device=device)

		# Penalizes predictions based on the human noise transition matrix
		if  self.N_h is  not  None:
			loss_h = F.nll_loss(torch.log(torch.matmul(softmax_pred, self.N_h)), n_input, reduction='mean')
			correction = torch.cat((correction, torch.tensor([loss_h]).to(device)))
		
		# Penalizes predictions based on the base model noise transition matrix
		if  self.N_b is  not  None:
			loss_b = F.nll_loss(torch.log(torch.matmul(softmax_pred, self.N_b)), n_input, reduction='mean')
			correction = torch.cat((correction, torch.tensor([loss_b]).to(device)))

		return  self.C * torch.mean(correction)

	def  forward(self, prediction, n_input, target):
		# Compute standard classification loss
		base_loss = self.loss1(prediction, target)

		# Compute noise correction term
		correction = self.noiseCorrection(prediction, n_input)

		# Total loss = base loss + noise correction
		return base_loss + correction
```

### 🔧 Basic Training Loop

```python
# Uses the CorrectionLoss that combines standard classification loss
# (CrossEntropy) with optional noise correction from annotator profiles.
criterion = CorrectionLoss(nn.CrossEntropyLoss(), 0.1, noise_H)
n_classes = 10 #CIFAR10N contains 10 classes

EPOCHS = 100
for _ in  range(EPOCHS):
	for i, (imgs, consensus, n_labels) in  enumerate(trainloader):
		imgs = imgs.to(device)
		consensus = consensus.to(device)
		n_labels = n_labels.to(device)
			
		# Convert noisy labels to one-hot vectors
		n_labels = F.one_hot(n_labels, num_classes=n_classes).to(device, dtype=torch.float32)

		# Forward pass
		optimizer.zero_grad()
		_, _, outputs = adapt_model(imgs, n_labels)

		# Loss calculation and backpropagation
		loss = criterion(outputs, torch.argmax(n_labels, dim=1), consensus)
		loss.backward()
		optimizer.step()

		# Compute Predictions
		_, prediction = torch.max(outputs, 1)
```