from torch.utils.data import Dataset

# Train Dataloader
class Train(Dataset):
    def __init__(self, data):
        self.data = data

        # TODO: ADD CODE HERE

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # TODO: ADD CODE HERE

        return index