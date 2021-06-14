from tensorflow import keras
import losses, train, models
import tensorflow_addons as tfa
import os

# basic_model = models.buildin_models("ResNet101V2", dropout=0.4, emb_shape=512, output_layer="E")
# basic_model = models.buildin_models("ResNest50", dropout=0.4, emb_shape=512, output_layer="E")
# basic_model = models.buildin_models('EfficientNetB4', dropout=0, emb_shape=256, output_layer="GDC")
# basic_model = mobile_facenet.mobile_facenet(256, dropout=0, name="mobile_facenet_256")
# basic_model = mobile_facenet.mobile_facenet(256, dropout=0, name="se_mobile_facenet_256", use_se=True)
#basic_model = models.buildin_models("MobileNet", dropout=0, emb_shape=256, output_layer="GDC")
model_name = "binarydensenet"
basic_model = models.buildin_models(model_name, dropout=0, emb_shape=256, output_layer="GDC")
dataset_name = 'ms1m-retinaface-t1'
data_path = os.path.join('.','datasets',dataset_name)
eval_paths = [os.path.join(data_path,bin_file) for bin_file in ['lfw.bin','cfp_fp.bin','agedb_30.bin']]

tt = train.Train(data_path, save_path='keras_'+model_name+'_'+dataset_name+'.h5', eval_paths=eval_paths,
                basic_model=basic_model, lr_base=0.1, batch_size=128, random_status=2)
#optimizer = tfa.optimizers.AdamW(weight_decay=5e-5)
optimizer = tfa.optimizers.SGDW(learning_rate=0.1, weight_decay=5e-4, momentum=0.9)

sch = [
  #{"loss": keras.losses.CategoricalCrossentropy(label_smoothing=0.1), "centerloss": 0.01, "optimizer": optimizer, "epoch": 20},
  #{"loss": keras.losses.CategoricalCrossentropy(label_smoothing=0.1), "centerloss": 0.1, "epoch": 20},
  #{"loss": keras.losses.CategoricalCrossentropy(label_smoothing=0.1), "centerloss": 0.5, "epoch": 20},
  #{"loss": losses.ArcfaceLoss(), "epoch": 20, "triplet": 64, "alpha": 0.3},
  #{"loss": losses.ArcfaceLoss(), "epoch": 20, "triplet": 64, "alpha": 0.25},
  {"loss": losses.ArcfaceLoss(scale=64), "epoch": 50,"optimizer":optimizer},
]
tt.train(sch, 0)