from easydict import EasyDict as edict

config = edict()
config.dataset = "webface"
config.embedding_size = 512
config.sample_rate = 1
config.fp16 = False
config.momentum = 0.9
config.weight_decay = 5e-4
config.batch_size = 383
config.lr = 0.1  # batch size is 512
config.output = "ms1mv3_arcface_r50_gpu3"
config.rec = "/home/adminuser/FaceRecognition/insightface/recognition/arcface_torch/_datasets_/train_tmp/faces_webface_112x112"
config.num_classes = 10572
config.num_image = "forget"
config.num_epoch = 34
config.warmup_epoch = -1
config.val_targets = ["lfw", "cfp_fp", "agedb_30"]

config.rank = 1
config.local_rank = 3

def lr_step_func(epoch):
    return ((epoch + 1) / (4 + 1)) ** 2 if epoch < config.warmup_epoch else 0.1 ** len([m for m in [20, 28, 32] if m - 1 <= epoch])
config.lr_func = lr_step_func