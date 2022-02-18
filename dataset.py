from sys import path
import torch
import matplotlib.pyplot as plt
import glob
from datasets.train_loader import FileLoader
from torch.utils.data import DataLoader
from run_utils.utils import check_manual_seed
import tqdm,os


def worker_init_fn(worker_id):
    # ! to make the seed chain reproducible, must use the torch random, not numpy
    # the torch rng from main thread will regenerate a base seed, which is then
    # copied into the dataloader each time it created (i.e start of each epoch)
    # then dataloader with this seed will spawn worker, now we reseed the worker
    worker_info = torch.utils.data.get_worker_info()
    # to make it more random, simply switch torch.randint to np.randint
    worker_seed = torch.randint(0, 2 ** 32, (1,))[0].cpu().item() + worker_id
    # print('Loader Worker %d Uses RNG Seed: %d' % (worker_id, worker_seed))
    # retrieve the dataset copied into this worker process
    # then set the random seed for each augmentation
    worker_info.dataset.setup_augmentor(worker_id, worker_seed)
    return


class TrainManager():
    """Either used to view the dataset or to initialise the main training loop."""

    def __init__(self,run_info):
        super().__init__()
        self.run_info = run_info
        self.train_dir_list = self.run_info['dir_path']['train']
        self.valid_dir_list = self.run_info['dir_path']['valid']
        self.debug = False
        self.type_classification = True
        self.shape_info = self.run_info['shape_info']
        ##Be cautious about this edit
        self.nr_gpus = 1 
        self.seed = 10
        return

    ####
    def view_dataset(self, mode="train"):
        """
        Manually change to plt.savefig or plt.show 
        if using on headless machine or not
        """
        self.nr_gpus = 1
        self.seed = 10
        check_manual_seed(self.seed)
        prep_func, prep_kwargs = self.run_info['target_info']['viz']
        dataloader = self._get_datagen()
        print(f'Length = {len(dataloader)}')
        for batch_data in tqdm.tqdm(dataloader):
            # convert from Tensor to Numpy
            batch_data = {k: v.numpy() for k, v in batch_data.items()}
            viz = prep_func(batch_data, is_batch=True, **prep_kwargs)
            plt.imshow(viz)
            plt.show()
        # self.nr_gpus = -1 #Dont know why
        return

    ####
    # def _get_datagen(self, batch_size, run_mode, target_gen, nr_procs=os.cpu_count):
    def _get_datagen(self, nr_procs=os.cpu_count()):
        nr_procs = nr_procs if not self.debug else 0
        print(f'Number of workers = {nr_procs},Debug mode is {self.debug}')

        # ! Hard assumption on file type
        file_list = []
        if self.run_info['mode'] == "train":
            data_dir_list = self.train_dir_list
        else:
            data_dir_list = self.valid_dir_list
        for dir_path in data_dir_list:
            file_list.extend(glob.glob("%s/*.npy" % dir_path))
        file_list.sort()  # to always ensure same input ordering
        assert len(file_list) > 0, (
            "No .npy found for `%s`, please check `%s` in `config.py`"
            % (self.run_info['mode'], "%s_dir_list" % self.run_info['mode'])
        )
        print("Dataset %s: %d" % (self.run_info['mode'], len(file_list)))
        input_dataset = FileLoader(
            file_list,
            mode=self.run_info['mode'],
            with_type=self.type_classification,
            setup_augmentor=nr_procs == 0,
            target_gen=self.run_info['target_info']['gen'],
            **self.shape_info[self.run_info['mode']]
        )
        dataloader = DataLoader(
            input_dataset,
            num_workers=nr_procs,
            batch_size=self.run_info['batch_size'][self.run_info['mode']] * self.nr_gpus,
            shuffle=True,
            drop_last=self.run_info['mode'] == "train",
            worker_init_fn=worker_init_fn,
        )
        return dataloader