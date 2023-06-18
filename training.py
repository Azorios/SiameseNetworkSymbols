import torch
from help_functions import show_plot, save_best_loss, load_best_loss


def training(train_loader, val_loader, device, optimizer, model, loss_fn):
    train_counter = []
    val_counter = []
    train_loss_history = []
    val_loss_history = []
    train_iteration_number = 0
    val_iteration_number = 0


    n_epochs = 500

    best_loss = load_best_loss()

    # Iterate through the epochs
    for epoch in range(n_epochs):
        # Training phase
        model.train()

        train_epoch_loss = 0.0

        # Iterate over batches
        for i, (img0, img1, label, class0, class1) in enumerate(train_loader, 0):

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
            # print(f"Epoch number {epoch}\n Current train loss {loss_contrastive.item()}\n")

            train_epoch_loss += loss_contrastive.item()

            train_loss_history.append(loss_contrastive.item())
            train_iteration_number += 1
            train_counter.append(train_iteration_number)

        # Validation phase
        model.eval()
        val_epoch_loss = 0.0

        with torch.no_grad():

            for i, (img0, img1, label, class0, class1) in enumerate(val_loader, 0):

                # Convert images and labels to the used device accordingly (cpu or cuda)
                img0, img1, label = img0.to(device), img1.to(device), label.to(device)

                # Pass in the two images into the network and obtain two outputs/vectors
                output1_val, output2_val = model(img0, img1)

                # Pass the outputs of the networks and label into the loss function
                loss_val = loss_fn(output1_val, output2_val, label)

                # Print out the loss
                # print(f"Epoch number {epoch}\n Current val loss {loss_val.item()}\n")

                val_epoch_loss += loss_val.item()

                val_loss_history.append(loss_val.item())
                val_iteration_number += 1
                val_counter.append(val_iteration_number)

        if loss_val < best_loss:
            best_loss = loss_val
            save_best_loss(best_loss)
            torch.save(model.state_dict(), "./output/model.pth")
            print(f"Model saved. New best loss: {best_loss}\n+")

        train_epoch_loss /= len(train_loader)
        val_epoch_loss /= len(val_loader)
        print(f"Epoch number {epoch}\n Train Loss {train_epoch_loss}\n Val Loss {val_epoch_loss}\n")

    show_plot(train_counter, val_counter, train_loss_history, val_loss_history, best_loss)

    return None
