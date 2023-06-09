import os
import paddle
import paddle.fluid as fluid
import paddle.nn.functional as F
from paddle.optimizer import lr
# from model import PPLCNet
from models import ResNet_basic111_maxpool
import random
import numpy as np
from dataset import AngleClass
from configs.config_base import Config
from visualdl import LogWriter
from loss import MultiLabelLoss
from metric import HammingDistance
import warnings

warnings.filterwarnings("ignore")
import logging

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '0'

visual_log = LogWriter("training_log_base")

seed = Config["solver"]["seed"]
paddle.seed(seed)
np.random.seed(seed)
random.seed(seed)

num_epoch = Config["solver"]["num_epoch"]
save_path = Config["model"]["save_path"]
MODEL_STAGES_PATTERN = {
    "PPLCNet": ["blocks2", "blocks3", "blocks4", "blocks5", "blocks6"]
}

train_dataset = AngleClass(Config, 'train', False)
train_loader = paddle.io.DataLoader(train_dataset, batch_size=Config["dataset"]["train_batch"], shuffle=True,
                                    num_workers=0)

test_dataset = AngleClass(Config, 'test', False)
test_loader = paddle.io.DataLoader(test_dataset, batch_size=Config["dataset"]["test_batch"], shuffle=False,
                                   num_workers=0)
print("train_dataset: ", len(train_dataset), "test_dataset: ", len(test_dataset))

# net = PPLCNet(scale=Config["model"]["scale"], stages_pattern=MODEL_STAGES_PATTERN["PPLCNet"])
net = ResNet_basic111_maxpool(Config["dataset"]["resize_pad"], 3, [(1, 32), (1, 64), (1, 64), (1, 64)], class_num=7)

ham_dic_min = 1000
schedule = lr.MultiStepDecay(learning_rate=Config['solver']['base_lr'], milestones=Config["solver"]["milestones"],
                             gamma=Config["solver"]["gamma"])
opt = paddle.optimizer.Adam(learning_rate=schedule, parameters=net.parameters())
save_dir = '/home/aistudio/work/PPLCNet/save_weight'
multi_label_loss = MultiLabelLoss()
ham_dic = HammingDistance()

with fluid.dygraph.guard():
    net.train()
    schedule.step()

    try:
        for epoch in range(0, Config["solver"]["num_epoch"]):

            net.train()
            schedule.step()
            for i, (img, labels) in enumerate(train_loader):
                predict = net(img)
                loss_dict = multi_label_loss(predict, labels)
                avg_loss = loss_dict['MultiLabelLoss']

                avg_loss.backward()
                opt.minimize(avg_loss)
                net.clear_gradients()

                if i % Config["solver"]["loss_print_freq"] == 0:
                    print("epoch {}/{} iter {} loss: {} ".format(epoch, num_epoch, i, avg_loss.numpy()))

            total, acc = 0, 0
            net.eval()
            model_state = net.state_dict()
            save_p = os.path.join(save_dir, 'temp_epoch_base')
            paddle.save(model_state, save_p)

            ham_dic.reset()

            for i, (img, labels) in enumerate(test_loader):
                predict = net(img)
                harm_metric = ham_dic.forward(predict, labels)
            harm_metric_val = harm_metric['HammingDistance'].numpy()[0]
            visual_log.add_scalar(tag="eval_score", step=epoch, value=harm_metric_val)
            print("epoch {} /{} harming_distance {}".format(epoch, num_epoch, harm_metric_val))

            # print(harm_metric_val , ham_dic_min,  harm_metric_val < ham_dic_min)

            if harm_metric_val < ham_dic_min:
                ham_dic_min = harm_metric_val
                model_state = net.state_dict()
                save_p = os.path.join(save_dir, 'base_best_model')
                paddle.save(model_state, save_p)
                print("min harming distance {}".format(ham_dic_min))


    except KeyboardInterrupt:
        model_state = net.state_dict()
        paddle.save(model_state, Config["model"]["interrupt_path"])
