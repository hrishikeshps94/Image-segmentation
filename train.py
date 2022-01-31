from collections import OrderedDict
import torch
import torch.nn.functional as F
from losses import cross_entropy_loss,dice_loss,mse_loss,msge_loss
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_step(batch_data,model,optimizer,lr_scheduler, loss_opts):
    # TODO: synchronize the attach protocol
    # run_info, state_info = run_info
    loss_func_dict = {
        "bce": cross_entropy_loss,
        "dice": dice_loss,
        "mse": mse_loss,
        "msge": msge_loss,
    }
    result_dict = {"EMA": {}}
    # use 'ema' to add for EMA calculation, must be scalar!
    track_value = lambda name, value: result_dict["EMA"].update({name: value})

    ####
    # model = run_info["net"]["desc"]
    # optimizer = run_info["net"]["optimizer"]

    ####
    imgs = batch_data["img"]
    true_np = batch_data["np_map"]
    true_hv = batch_data["hv_map"]

    imgs = imgs.to(device).type(torch.float32)  # to NCHW
    imgs = imgs.permute(0, 3, 1, 2).contiguous()

    # HWC
    true_np = true_np.to(device).type(torch.int64)
    true_hv = true_hv.to(device).type(torch.float32)

    true_np_onehot = (F.one_hot(true_np, num_classes=2)).type(torch.float32)
    true_dict = {
        "np": true_np_onehot,
        "hv": true_hv,
    }

    # if model.module.nr_types is not None:
    if model.nr_types is not None:
        true_tp = batch_data["tp_map"]
        true_tp = torch.squeeze(true_tp).to(device).type(torch.int64)
        # true_tp_onehot = F.one_hot(true_tp, num_classes=model.module.nr_types)
        true_tp_onehot = F.one_hot(true_tp, num_classes=model.nr_types)
        true_tp_onehot = true_tp_onehot.type(torch.float32)
        true_dict["tp"] = true_tp_onehot

    ####
    model.train()
    model.zero_grad()  # not rnn so not accumulate

    pred_dict = model(imgs)
    pred_dict = OrderedDict(
        [[k, v.permute(0, 2, 3, 1).contiguous()] for k, v in pred_dict.items()]
    )
    pred_dict["np"] = F.softmax(pred_dict["np"], dim=-1)
    # if model.module.nr_types is not None:
    if model.nr_types is not None:
        pred_dict["tp"] = F.softmax(pred_dict["tp"], dim=-1)

    ####
    loss = 0
    # loss_opts = run_info["net"]["extra_info"]["loss"]
    for branch_name in pred_dict.keys():
        for loss_name, loss_weight in loss_opts[branch_name].items():
            loss_func = loss_func_dict[loss_name]
            loss_args = [true_dict[branch_name], pred_dict[branch_name]]
            if loss_name == "msge":
                loss_args.append(true_np_onehot[..., 1])
            term_loss = loss_func(*loss_args)
            track_value("loss_%s_%s" % (branch_name, loss_name), term_loss.cpu().item())
            loss += loss_weight * term_loss

    track_value("overall_loss", loss.cpu().item())
    # * gradient update

    # torch.set_printoptions(precision=10)
    loss.backward()
    optimizer.step()
    lr_scheduler.step()
    ####

    # # pick 2 random sample from the batch for visualization
    # sample_indices = torch.randint(0, true_np.shape[0], (2,))

    # imgs = (imgs[sample_indices]).byte()  # to uint8
    # imgs = imgs.permute(0, 2, 3, 1).contiguous().cpu().numpy()

    # pred_dict["np"] = pred_dict["np"][..., 1]  # return pos only
    # pred_dict = {
    #     k: v[sample_indices].detach().cpu().numpy() for k, v in pred_dict.items()
    # }

    # true_dict["np"] = true_np
    # true_dict = {
    #     k: v[sample_indices].detach().cpu().numpy() for k, v in true_dict.items()
    # }

    # # * Its up to user to define the protocol to process the raw output per step!
    # result_dict["raw"] = {  # protocol for contents exchange within `raw`
    #     "img": imgs,
    #     "np": (true_dict["np"], pred_dict["np"]),
    #     "hv": (true_dict["hv"], pred_dict["hv"]),
    # }
    return result_dict


def valid_step(batch_data,model, loss_opts):
    ####
    loss_func_dict = {
        "bce": cross_entropy_loss,
        "dice": dice_loss,
        "mse": mse_loss,
        "msge": msge_loss,
    }
    result_dict = {"Scalar": {}}
    # use 'ema' to add for EMA calculation, must be scalar!
    track_value = lambda name, value: result_dict["Scalar"].update({name: value})
    model.eval()  # infer mode

    ####
    imgs = batch_data["img"]
    true_np = batch_data["np_map"]
    true_hv = batch_data["hv_map"]

    imgs_gpu = imgs.to(device).type(torch.float32)  # to NCHW
    imgs_gpu = imgs_gpu.permute(0, 3, 1, 2).contiguous()

    # HWC
    true_np = true_np.to(device).type(torch.int64)
    true_hv = true_hv.to(device).type(torch.float32)
    disp_true_np = torch.squeeze(batch_data["np_map"]).to(device).type(torch.int64)
    disp_true_hv = torch.squeeze(batch_data["hv_map"]).to(device).type(torch.float32)
    true_np_onehot = (F.one_hot(true_np, num_classes=2)).type(torch.float32)

    disp_true_dict = {
        "np": disp_true_np,
        "hv": disp_true_hv,
    }
    true_dict = {
        "np": true_np_onehot,
        "hv": true_hv,
    }

    if model.nr_types is not None:
        true_tp = batch_data["tp_map"]
        true_tp = torch.squeeze(true_tp).to(device).type(torch.int64)
        disp_true_dict["tp"]=true_tp
        true_tp_onehot = F.one_hot(true_tp, num_classes=model.nr_types)
        true_tp_onehot = true_tp_onehot.type(torch.float32)
        true_dict["tp"] = true_tp_onehot

    # --------------------------------------------------------------
    with torch.no_grad():  # dont compute gradient
        org_dict = model(imgs_gpu)
        pred_dict = OrderedDict(
        [[k, v.permute(0, 2, 3, 1).contiguous()] for k, v in org_dict.items()] )       
        disp_pred_dict = OrderedDict(
            [[k, v.permute(0, 2, 3, 1).contiguous()] for k, v in org_dict.items()]
        )
        pred_dict["np"] = F.softmax(pred_dict["np"], dim=-1)
        disp_pred_dict["np"] = F.softmax(disp_pred_dict["np"], dim=-1)[..., 1]
        if model.nr_types is not None:
            pred_dict["tp"] = F.softmax(pred_dict["tp"], dim=-1)
            type_map = F.softmax(disp_pred_dict["tp"], dim=-1)
            type_map = torch.argmax(type_map, dim=-1, keepdim=False)
            type_map = type_map.type(torch.float32)
            disp_pred_dict["tp"] = type_map
        ####
        loss = 0
        # loss_opts = run_info["net"]["extra_info"]["loss"]
        for branch_name in pred_dict.keys():
            for loss_name, loss_weight in loss_opts[branch_name].items():
                loss_func = loss_func_dict[loss_name]
                loss_args = [true_dict[branch_name], pred_dict[branch_name]]
                if loss_name == "msge":
                    loss_args.append(true_np_onehot[..., 1])
                term_loss = loss_func(*loss_args)
                track_value("loss_%s_%s" % (branch_name, loss_name), term_loss.cpu().item())
                loss += loss_weight * term_loss

        track_value("overall_loss", loss.cpu().item())
    # * Its up to user to define the protocol to process the raw output per step!
    result_dict['raw'] = {  # protocol for contents exchange within `raw` {
            "imgs": imgs.numpy(),
            "true_np": disp_true_dict["np"].cpu().numpy(),
            "true_hv": disp_true_dict["hv"].cpu().numpy(),
            "prob_np": disp_pred_dict["np"].cpu().numpy(),
            "pred_hv": disp_pred_dict["hv"].cpu().numpy(),
        }
    if model.nr_types is not None:
        result_dict["raw"]["true_tp"] = disp_true_dict["tp"].cpu().numpy()
        result_dict["raw"]["pred_tp"] = disp_pred_dict["tp"].cpu().numpy()
    return result_dict