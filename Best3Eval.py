from stable_baselines3.common.callbacks import EvalCallback
import os

class BestThreeEvalCallback(EvalCallback):
    def __init__(self, *args, **kwargs):
        super(BestThreeEvalCallback, self).__init__(*args, **kwargs)
        self.best_models = []
        self.if_save = False

    def _on_step(self) -> bool:
        result = super(BestThreeEvalCallback, self)._on_step()
        if len(self.best_models) < 10:
            self.best_models.append((self.best_mean_reward, self.model.get_parameters()))
            self.best_models.sort(key=lambda x: x[0], reverse=True)
        elif self.best_mean_reward > self.best_models[-1][0]:
            self.best_models[-1] = (self.best_mean_reward, self.model.get_parameters())
            self.best_models.sort(key=lambda x: x[0], reverse=True)
            self.if_save = True
        
        # Save the best models
        if self.if_save:
            # for i, (reward, params) in enumerate(self.best_models):
            #     model_path = os.path.join(self.best_model_save_path, f"best_model_{reward}.zip")
            #     self.model.set_parameters(params)
            #     self.model.save(model_path)
            # 保存当前模型
            model_path = os.path.join(self.best_model_save_path, f"best_model_{self.best_mean_reward}.zip")
            self.model.save(model_path)
            self.if_save = False
        
        return result

# # 使用自定义的回调类
# eval_callback = BestThreeEvalCallback(eval_env, best_model_save_path=log_path,
#                                       log_path=log_path, eval_freq=10000,
#                                       deterministic=True, render=False)

# model = PPO("MlpPolicy", train_env, verbose=1, device='cpu', tensorboard_log="./logs/tensorboard/")
# model.learn(total_timesteps=2500_000, progress_bar=True, callback=[eval_callback])