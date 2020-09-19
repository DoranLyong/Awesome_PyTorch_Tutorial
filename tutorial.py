from libs.data.fashion_mnist import * 
from libs.utils.figure import plt 



"""
1. DataLoader 
"""
train_iter, test_iter = load_data_fashion_mnist()

imgs = [] 
lbls = [] 

for idx, data in enumerate(test_iter):
    if(idx>=0 and idx<10):
        imgs.append(data[0])
        lbls.append(data[1])

    if (idx>=10):
        break 

show_fashion_mnist(imgs, get_fashion_mnist_labels(lbls))


"""
2. Build_model + optim + lr_scheduler 
"""


"""
3. Build_engine : 

"""


"""
4. Run training 

"""


"""
5. Testing
"""


