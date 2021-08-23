from datetime import datetime
import os


def save_model(model, i=-1, name='model', model_path='model_record/'):
    # save the model
    output_path = "model_record/"
    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    print("[INFO] {} - Saving model ...".format(now))
    # try:
    #     os.mkdir(logdir)
    # except OSError as err:
    #     # print("Creation of directory {} failed:{}".format(logdir, err))
    if i == -1:
        model.save(output_path + name + ".hdf5")
    else:
        model.save(model_path + "{}_".format(i) + name + ".hdf5")
