import numpy as np
from config import run_info
from run_utils.utils import get_model_summary
from dataset import TrainManager
import tqdm,torch
from train import train_step,valid_step
from collections import Counter
import wandb,os
from visualize import proc_valid_step_output
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Run():
    def __init__(self,run_info):
        self.run_info = run_info
        self.model = run_info['net']()
        print(get_model_summary(self.model, (3, 256, 256), device='cpu'))
        self.optimizer = self.run_info["optimizer"](params = self.model.parameters(),lr=1.0e-4)
        self.curent_epoch = 0
        self.load_dataset()
        self.restore_model()
        self.lr_scheduler = self.run_info['lr_scheduler'](self.optimizer)
        self.lowest_loss = np.inf
        
        

    def load_dataset(self):
        self.train_dataloader = TrainManager(self.run_info)._get_datagen()
        self.run_info.update({'mode':'valid'})
        self.val_dataloader = TrainManager(self.run_info)._get_datagen()
        return


    def restore_model(self):
        checkpoint_filename = os.path.join(self.run_info['save_path'], 'last.pth')
        if not os.path.exists(checkpoint_filename):
            print("Couldn't find checkpoint file. Starting training from the beginning.")
            self.model.to(device)
            return
        data = torch.load(checkpoint_filename)
        self.model.load_state_dict(data['generator_state_dict'])
        self.model.to(device)
        self.optimizer.load_state_dict(data['optimizer_state_dict'])
        global_step = data['step']
        self.curent_epoch = global_step//len(self.train_dataloader)
        print(f"Restored model at step {self.curent_epoch}.")
        return

    def save_checkpoint(self,global_step,type='last'):
        checkpoint_folder = self.run_info['save_path']
        if not os.path.exists(checkpoint_folder):
            os.makedirs(checkpoint_folder)
        checkpoint_filename = os.path.join(checkpoint_folder, f'{type}.pth')
        save_data = {
            'step': global_step,
            'generator_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lowest_loss':self.lowest_loss
        }
        torch.save(save_data, checkpoint_filename)
        return


    def train(self):
        for epoch in range(self.curent_epoch,self.run_info['nr_epochs']):
            epoch_stat = {"EMA": {}}
            val_epoch_stat = {"Scalar": {}}
            for count,batch_data in tqdm.tqdm(enumerate(self.train_dataloader)):
                step_stat = train_step(batch_data,self.model,self.optimizer,self.lr_scheduler,self.run_info["loss"])
                epoch_stat['EMA'] = dict(Counter(epoch_stat['EMA'])+Counter(step_stat['EMA']))
            final_epoch_stat = {k:v/count+1 for k,v in epoch_stat['EMA'].items()}
            # if final_epoch_stat['overall_loss']<self.lowest_loss:
            #     self.save_checkpoint(epoch*count,'train_best')

            self.save_checkpoint(epoch*count,'last')         
            for loss_name,loss_val in final_epoch_stat.items():
                wandb.log({'train_'+loss_name:loss_val})
                wandb.log({'learningrate':self.optimizer.param_groups[0]['lr']})
            print(f' Epoch {epoch} losses = {final_epoch_stat}')
            for val_count,batch_data in tqdm.tqdm(enumerate(self.val_dataloader)):
                val_result_dict = valid_step(batch_data,self.model,self.run_info["loss"])
                val_epoch_stat['Scalar'] = dict(Counter(val_epoch_stat['Scalar'])+Counter(val_result_dict['Scalar']))
            final_val_epoch_stat = {k:v/val_count+1 for k,v in val_epoch_stat['Scalar'].items()}
            if final_val_epoch_stat['overall_loss']<self.lowest_loss:
                self.save_checkpoint(epoch*count,'best')
            for loss_name,loss_val in final_val_epoch_stat.items():
                wandb.log({'val_'+loss_name:loss_val})
            visualise_val = proc_valid_step_output(val_result_dict['raw'],self.run_info['batch_size']['valid'],self.run_info['nr_class'])        
            for val_image_name,val_image in visualise_val['image'].items():
                wandb.log({val_image_name:wandb.Image(val_image)})
        return

def main(run_info):
  trainer = Run(run_info)
  trainer.train()


if __name__ == "__main__":
    wandb.init(project="ISBI_Hover",name='hover_log')
    main(run_info)