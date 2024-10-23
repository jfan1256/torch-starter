## Pytorch Template
This is a Pytorch Template that you can use to train your models. 

## Table of Content
✅ Tutorial

✅ Things to Do

✅ Features

## Tutorial
1. Clone the repository using: ```git clone [repository link]```
2. Install the requirements using: ```pip install -r requirements.txt```
3. Install the anaconda pytorch (not required): ```conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia```
3. Install the setup.py using: ```pip install -e .```

## Things to Do
Every instance of `# TODO: ADD CODE HERE` should be replaced with your custom code. 
For example, in `./class_model/model.py`, you should replace `# TODO: ADD CODE HERE` with your model implementation.

**Notable tasks:**
1. `./class_model/model.py`: Add your model here.
2. `./class_dataloder/dataloader.py`: Add your dataloader here.
3. `./exec_train/utils_train.py`: Add your training procedure here.
4. `./configs/train/train.yaml`: Add your configurations here.

## Features
1. **Metric Logger**: A prebuilt system that beautifully displays training and validation losses during each epoch, providing real-time insights into model performance.
2. **Model Checkpointing and Loss Logging**: Automatic saving of model checkpoints and loss statistics after each epoch, ensuring that your progress is tracked seamlessly.
3. **Configurable Settings**: The code is designed to read all configurations from a YAML file, allowing for easy adjustments without modifying the code.
4. **Multi-GPU Support**: Built-in support for multi-GPU training. Simply configure the parameters in `./configs/train/train.yaml` to enable distributed training.
5. **Early Stopping**: Integrated early-stopping functionality to halt training when the model stops improving, preventing overfitting.
6. **Learning Rate Scheduler**: A pre-configured learning rate scheduler is ready to use, managed through `./exec_train/utils_eta.py`.
7. **Loss Plotting**: After training, the code generates a clear and visually appealing plot of the training and validation losses for easy analysis.
