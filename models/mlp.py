import torch.nn as nn

class MLPClassifier(nn.Module):
    def __init__(self, input_dim=139, num_classes=5):
        super(MLPClassifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        return self.fc(x)
    
def mlp(**kwargs):
    model = MLPClassifier(**kwargs)
    return model