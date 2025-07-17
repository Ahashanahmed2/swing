from stable_baselines3.common.callbacks import BaseCallback
import pandas as pd
import os
from datetime import datetime
import shutil

class LossLoggingCallback(BaseCallback):
    def __init__(self, log_path="./csv/loss_log.csv", backup_path="./output/ai_signal/loss_log.csv", verbose=1):
        super().__init__(verbose)
        self.log_path = log_path
        self.backup_path = backup_path
        self.logs = []

        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        os.makedirs(os.path.dirname(backup_path), exist_ok=True)

    def _on_step(self) -> bool:
        # ✅ Always return True
        loss = self.model.logger.name_to_value.get('train/loss')
        if loss is not None:
            log_entry = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'timesteps': self.num_timesteps,
                'loss': loss,
                'learning_rate': self.model.lr_schedule(self.num_timesteps),
                'exploration_rate': self.model.exploration_rate,
            }
            self.logs.append(log_entry)

        return True  # ✅ Required by BaseCallback

    def _on_training_end(self):
        # Append to existing CSV if it exists
        if os.path.exists(self.log_path):
            old_df = pd.read_csv(self.log_path)
            new_df = pd.DataFrame(self.logs)
            df = pd.concat([old_df, new_df], ignore_index=True)
        else:
            df = pd.DataFrame(self.logs)

        df.to_csv(self.log_path, index=False)
        shutil.copyfile(self.log_path, self.backup_path)

        if self.verbose:
            print(f"✅ Loss log saved to {self.log_path}")
            print(f"📁 Backup saved to {self.backup_path}")
