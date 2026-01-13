# L2CU: Learning To Complement Unseen Users
<a href="https://ieeexplore.ieee.org/document/11314492">
  <img src="./assets/ieee_logo.png" alt="IEEE Logo" width="50"/>
</a>
<a href="https://arxiv.org/abs/2601.06119">
  <img src="./assets/arxiv_logo.jpg" alt="Arxiv Logo" width="50"/>
</a>


- Below are basic code snippets that illustrate the core components of the framework, including:
  - Model definition  
  - Loss function  
  - Data loader  
  - Training loop
- A complete code example based on the CIFAR10N experiment is provided in the `codes/` directory that goes through Python notebooks 1 to 5.
- `code/matrics.py` contains the implementations of the evaluation criteria.

### Basic Model architecture
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
			n_classes = 10
			
			# base model
			# This can be any model architecture of choice that supports classification. The output should be the classification head logits
			self.base_model = 
			
			# human label encoder 
			self.n_l_encoder = nn.Sequential(
			nn.Linear(n_classes, 32),
			nn.ReLU(),
			nn.Linear(32, n_classes)
			)  
			
			# decision model
			self.decision_ai = nn.Sequential(
			nn.Linear(2*n_classes, 64),
			nn.ReLU(),
			nn.Linear(64, 32),
			nn.ReLU(),
			nn.Linear(32, n_classes),
			)
		
		def  forward(self, imgs, n_l):
			img_features = self.base_model(imgs)
			n_l_features = self.n_l_encoder(n_l)
			out = torch.cat((img_features, n_l_features), dim=1)
			out = self.decision_ai(out)
			return img_features, n_l, out
```

### Basic data loader
> **Note:** Data are loaded per annotator profile

```python
    class  CIFAR10(VisionDataset):
	    def  __init__(self):
		    # data loading and etc
	    
	    def  __len__(self) -> int:
		    return  len(self.noisyLabels)
	    
	    def  __getitem__(self, index: int) -> Tuple[Any, Any]:
		    # data loading covers all the noisy labels from the annotator profile. Image and consensus label loading is adjusted accordingly.
		    img, consensus, n_l = self.data[index % len(self.data)], self.consensus[index % len(self.data)], self.noisyLabels[index]
		    
			### image transforms
	
		    # should return -> image, its consensus label, one of its noisy labels from the annotator profile
		    return img, consensus, n_l
```

### Loss function
```python
    class  CorrectionLoss(nn.Module):
    	def  __init__(self, loss1, C=0, N_human=None, N_base=None):
    		super().__init__()
    		self.loss1 = loss1 # a classification loss (cross entropy)
    		self.C = C
    		self.N_h = N_human # noise transition matrix for human
    		self.N_b = N_base. # noise transition matrix for the base model

		def  noiseCorrection(self, prediction, n_input):
			softmax_pred = F.softmax(prediction, dim=1)
			correction = torch.tensor([]).to(device=device)

			if  self.N_h is  not  None:
				loss_h = F.nll_loss(torch.log(torch.matmul(softmax_pred, self.N_h)), n_input, reduction='mean')
				correction = torch.cat((correction, torch.tensor([loss_h]).to(device)))

			if  self.N_b is  not  None:
				loss_b = F.nll_loss(torch.log(torch.matmul(softmax_pred, self.N_b)), n_input, reduction='mean')
				correction = torch.cat((correction, torch.tensor([loss_b]).to(device)))

			return  self.C * torch.mean(correction)

		def  forward(self, prediction, n_input, target):
			l = self.loss1(prediction, target)
			correction = self.noiseCorrection(prediction, n_input)
			return l+correction
```

### Basic training loop

```python
    criterion = CorrectionLoss(nn.CrossEntropyLoss(), 0.1, noise_H)
    n_classes = 10 #for CIFAR10N

    EPOCHS = 100
	for _ in  range(EPOCHS):
		for i, (imgs, consensus, n_labels) in  enumerate(trainloader):
			imgs = imgs.to(device)
			consensus = consensus.to(device)
			n_labels = n_labels.to(device)
				
			# making the noisy label a one-hot vector
			n_labels = F.one_hot(n_labels, num_classes=n_classes).to(device, dtype=torch.float32)
	
			# forward pass
			optimizer.zero_grad()
			_, _, outputs = adapt_model(imgs, n_labels)
	
			# loss calculation
			loss = criterion(outputs, torch.argmax(n_labels, dim=1), consensus)
			loss.backward()
			optimizer.step()
	
			_, prediction = torch.max(outputs, 1)
```