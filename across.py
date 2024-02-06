from pyod.models.cblof import CBLOF
from pyod.models.lmdd import LMDD
from pyod.models.cd import CD
from pyod.models.hbos import HBOS
from pyod.models.inne import INNE #then try w/o INNE
from pyod.models.lunar import LUNAR
from pyod.models.kpca import KPCA
from pyod.models.lof import LOF
from sklearn.preprocessing import robust_scale, scale, minmax_scale
import pandas as pd

def across_od(data): #where data is a numpy array 
    scores = [] #leave this blank
    model_list = [CBLOF(alpha = 0.75, beta = 3.0), CBLOF(alpha = 0.75, beta = 3.0), LMDD(), LOF(), LUNAR()] #fill with a list of model instances
    normalizations_list = [robust_scale, scale, minmax_scale, robust_scale, scale]
    for i, model in enumerate(model_list):
        X = normalizations_list[i](data)
        #fit/predict w/ the model
        pred = model.fit_predict(data)

        #append predictions to a list
        scores.append(pred)
    
    #convert list of lists (scores) into a dataframe. Each column is a models predictions
    scores_df = pd.DataFrame(scores).T

    #take the mean of each row to find "average" OD value
    mean_scores = scores_df.max(axis=1)

    print("CHECK")
    print(len(mean_scores))
    print(len(mean_scores.dropna()))

    #classify as an outlier (0 if <= 0.5, 1 if > 0.5)
    final_scores = mean_scores.apply(lambda x: 0 if x <= 0.5 else 1).to_list()

    return {
        'decisions': final_scores,
        'levels': mean_scores
    }