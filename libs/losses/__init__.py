"""
codesource : https://github.com/KaiyangZhou/deep-person-reid/blob/master/torchreid/losses/__init__.py
"""

from .cross_entropy_loss import CrossEntropyLoss



def DeepSupervision(criterion, outputs, y): 

    """
    Apply criterion to each element in a list. 

    Args: 
        * criterion : loss function 
        * outputs : tuple of model ouputs
        * y : ground truth 
    """
    loss = 0. 

    for out in outputs: 
        loss += criterion(out, y) 

    loss /= len(outputs)  # average_loss (=batch_loss)
    return loss 

