import os

# Choose one of the following "mnist", "cifar10", "cifar100"
datasets = ['cifar10'] 

# Choose optimizers from AdaLip, AdamLip, RMSLip
#                         AdaLip_U, AdamLip_U, RMSLip_U
# or any other the is implemented in tensorflow (adam, sgd, rmsprop)
optimizers = ["AdamLip"]

# number of runs
iterations = 1

# different learning rates
lrs = ["0.001"]



for dataset in datasets:
    for opt in optimizers:
        for lr in lrs:
            for it in range(iterations):
                os.system("python run_all.py " + dataset + " " + opt + " " + lr + " " + str(it) )
    

