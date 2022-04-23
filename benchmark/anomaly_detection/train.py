from toycar_library import *


# Check CUDA availability
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Training on:", device)

# Define configuration variables
config = {
  # directories
  "dev_directory": "./dev_data",
  "eval_directory": "./eval_data",
  "model_directory": "./model",
  "result_directory": "./result",
  "result_file": "result.csv",

  # audio parameters
  'n_mels': 128,
  'frames': 5,
  'n_fft': 1024,
  'hop_length': 512,
  'power': 2.0,

  # data
  "batch_size": 512,
  "num_workers": 2,
  "val_split": 0.1,
  # training
  "n_epochs": 100,
  "lr": 0.001
}

# check mode
# "development": mode == True
# "evaluation": mode == False
mode = com.command_line_chk()
if mode is None:
  sys.exit(-1)

# make output directory
os.makedirs(config["model_directory"], exist_ok=True)

# load base_directory list
dirs = com.select_dirs(param=config, mode=mode)

# loop of the base directory
for idx, target_dir in enumerate(dirs):
  print("\n===========================")
  print("[{idx}/{total}] {dirname}".format(dirname=target_dir, idx=idx + 1, total=len(dirs)))

  # set path
  machine_type = os.path.split(target_dir)[1]
  model_file_path = "{model}/model_{machine_type}.hdf5".format(model=config["model_directory"],machine_type=machine_type)

  if os.path.exists(model_file_path):
    com.logger.info("model exists")
    continue

  # generate dataset
  print("============== DATASET_GENERATOR ==============")

  # Import benchmark dataset
  train_val_set = get_benchmark(target_dir, config)

  # Define training, validation and test dataloader
  trainLoader, valLoader = get_dataloaders(config, train_val_set)

  print(len(train_val_set), len(trainLoader.dataset), len(valLoader.dataset))
  print("len train dataloader", len(trainLoader))
  print("len val dataloader", len(valLoader))

  # train model
  print("============== MODEL TRAINING ==============")
  net = AutoEncoder(config["n_mels"]*config["frames"])
  if torch.cuda.is_available():
    net = net.cuda()

  # Define the optimizer, the loss and the number of epochs
  criterion = nn.MSELoss()
  optimizer = optim.Adam(net.parameters())
  checkpoint = CheckPoint('./checkpoints', net, optimizer, 'max')

  # Training loop
  for epoch in range(config['n_epochs']):
    metrics = train_one_epoch(epoch, net, criterion, optimizer, trainLoader, valLoader, device)
    #checkpoint(epoch, metrics['val_acc'])

  sys.exit(-1)



# Retrieve best checkpoint and test the model
checkpoint.load_best()
checkpoint.save('final_best.ckp')
test_loss, test_acc = evaluate(net, criterion, testLoader, device)
print("Test Set Accuracy:", test_acc.get())