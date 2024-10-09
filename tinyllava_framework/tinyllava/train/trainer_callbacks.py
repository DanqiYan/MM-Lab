import os
import transformers
import wandb

from tinyllava.utils import log
from tinyllava.model.reft_model_shell import ReftModelShell

class SaveCallback(transformers.TrainerCallback):
    def __init__(self, trainer, training_recipe):
        self.trainer = trainer
        self.training_recipe = training_recipe

    def on_epoch_end(self,
                        args: transformers.TrainingArguments,
                        state: transformers.TrainerState,
                        control: transformers.TrainerControl,
                        **kwargs):
        if state.epoch % self.trainer.args.epoch_to_save == 0:
            log('saving checkpoint ...')
            self.trainer.args.output_dir = os.path.join(self.trainer.args.output_dir, "epoch_"+str(state.epoch))
            self.training_recipe.save(self.trainer.model, self.trainer)
            self.trainer.args.output_dir = os.path.dirname(self.trainer.args.output_dir)


class WandbLogCallback(transformers.TrainerCallback):
    def __init__(self, model):
        self.model = model

    def log_model_size_into_wandb(self):
        model = self.model
        total_params_llm = sum(p.numel() for p in model.language_model.parameters())
        total_params_vt = sum(p.numel() for p in model.vision_tower.parameters())
        total_params_connector = sum(p.numel() for p in model.connector.parameters())
        total_trainable_params_llm = sum(p.numel() for p in model.language_model.parameters() if p.requires_grad)
        total_trainable_params_vt = sum(p.numel() for p in model.vision_tower.parameters() if p.requires_grad)
        total_trainable_params_connector = sum(p.numel() for p in model.connector.parameters() if p.requires_grad)
        
        table = wandb.Table(columns=["component", "num_param"])

        table.add_data("total_params_llm", total_params_llm)
        if isinstance(model.language_model, ReftModelShell):
            total_params_llm_without_reft = sum(p.numel() for p in model.language_model.model.parameters())
            table.add_data("total_params_llm_without_reft", total_params_llm_without_reft)
            table.add_data("total_params_reft", total_params_llm - total_params_llm_without_reft)
        table.add_data("total_params_vt", total_params_vt)
        table.add_data("total_params_connector", total_params_connector)
        table.add_data("trainable_params_llm", total_trainable_params_llm)
        table.add_data("trainable_params_vt", total_trainable_params_vt)
        table.add_data("trainable_params_connector", total_trainable_params_connector)

        for name, param in model.named_parameters():
            if param.requires_grad:
                table.add_data(f"{name}", param.numel())

        wandb.log({"Model Parameters": table})

    def on_train_begin(self,
                       args: transformers.TrainingArguments,
                       state: transformers.TrainerState,
                       control: transformers.TrainerControl,
                       **kwargs):
        if args.local_rank == 0 and "wandb" in args.report_to:  # Only log from the main process
            self.log_model_size_into_wandb()