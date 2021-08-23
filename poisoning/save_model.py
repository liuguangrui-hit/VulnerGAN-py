from datetime import datetime
import os


def save_model(model, i=0, name='model', model_path='model_record/'):
    # save the model
    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    print("[INFO] {} - Saving model ...".format(now))
    if i == 0:
        model.save(model_path + "NIDS_" + name + ".hdf5")
    else:
        model.save(model_path + "{}_NIDS_".format(i) + name + ".hdf5")
