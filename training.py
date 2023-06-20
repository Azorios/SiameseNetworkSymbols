import torch
from help_functions import show_plot_simple


def training(train_loader, device, optimizer, model, loss_fn):
    counter = []
    loss_history = []
    iteration_number = 0

    model.train()

    # Iterate through the epochs
    for epoch in range(1000):

        # Iterate over batches
        for i, (img0, img1, label, _, _) in enumerate(train_loader, 0):

            # Convert images and labels to the used device accordingly (cpu or cuda)
            img0, img1, label = img0.to(device), img1.to(device), label.to(device)

            # Zero the gradients
            optimizer.zero_grad()

            # Pass in the two images into the network and obtain two outputs/vectors
            output1, output2 = model(img0, img1)

            # Pass the outputs of the networks and label into the loss function
            loss_contrastive = loss_fn(output1, output2, label)

            # Calculate the backpropagation
            loss_contrastive.backward()

            # Optimize
            optimizer.step()

            # Print out the loss
            print(f"Epoch number {epoch}\n Current loss {loss_contrastive.item()}\n")
            iteration_number += 1

            counter.append(iteration_number)
            loss_history.append(loss_contrastive.item())

    torch.save(model.state_dict(), "./output5/model.pth")
    show_plot_simple(counter, loss_history)

    return None
