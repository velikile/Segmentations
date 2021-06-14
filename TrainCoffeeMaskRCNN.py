import torch
from engine import train_one_epoch, evaluate
import utils
from CoffeeBeanDataSet import CoffeeBeanDataSet 
from ModelDef import get_model_is

def main():
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # our dataset has two classes only - background and person
    num_classes = 2
    batch_size = 10
    # use our dataset and defined transformations
    dataset = CoffeeBeanDataSet('CoffeeBeanTrain')
    dataset_test = CoffeeBeanDataSet('CoffeeBeanValidation') 

    # split the dataset in train and test set
    indices = torch.randperm(len(dataset)).tolist()
    #print(len(dataset),len(dataset_test))
    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=1,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=batch_size, shuffle=False, num_workers=1,
        collate_fn=utils.collate_fn)

    # get the model using our helper function
    model = get_model_is(num_classes)

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)
    num_epochs = 1

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)


    torch.save(model,'mrcnn')
if(__name__ == '__main__'):
    main()
