import model_params
import plugins

        
class CVPRTopBandit(plugins.Bandit):
    param_gen = dict(
            slm=model_params.cvpr_top,
            comparison='mult',
            )
             
class FG11TopBandit(plugins.Bandit):
    param_gen = dict(
            slm=model_params.fg11_top,
            comparison='mult',
            )
        


        
                
        
