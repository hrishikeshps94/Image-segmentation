import cc_torch
import torch
from skimage import morphology as morph
from misc.utils import center_pad_to_shape, cropping_center, get_bounding_box
import torchvision.transforms as T
from scipy.ndimage import measurements

def fix_mirror_padding(ann):
    """Deal with duplicated instances due to mirroring in interpolation
    during shape augmentation (scale, rotation etc.).
    
    """
    ann = torch.from_numpy(ann).to('cuda:0')
    current_max_id = torch.max(ann)
    inst_list = list(torch.unique(ann))
    inst_list.remove(0)  # 0 is background
    for inst_id in inst_list:
        inst_map = (ann == inst_id).type(torch.uint8)
        remapped_ids = cc_torch.connected_components_labeling(inst_map)
        # remapped_ids,num_feat = measurements.label(inst_map)
        remapped_ids[remapped_ids > 1] += current_max_id
        ann[remapped_ids > 1] = remapped_ids[remapped_ids > 1]
        current_max_id = torch.max(ann)
    return ann

def get_bounding_box(img):
    """Get bounding box coordinate information."""
    rows = torch.any(img, axis=1)
    cols = torch.any(img, axis=0)
    rmin, rmax = torch.where(rows)[0][[0, -1]]
    cmin, cmax = torch.where(cols)[0][[0, -1]]
    # due to python indexing, need to add 1 to max
    # else accessing will be 1px in the box, not out
    rmax += 1
    cmax += 1
    return [rmin, rmax, cmin, cmax]

def gen_instance_hv_map(ann, crop_shape):
    """Input annotation must be of original shape.
    
    The map is calculated only for instances within the crop portion
    but based on the original shape in original image.

    Perform following operation:
    Obtain the horizontal and vertical distance maps for each
    nuclear instance.

    """
    orig_ann = ann.copy()  # instance ID map
    fixed_ann = fix_mirror_padding(orig_ann)
    # re-cropping with fixed instance id map
    crop_ann = cropping_center(fixed_ann, crop_shape) # UPDATED THIS DON'T KNOW THE REPERCUSSIONS
    # crop_ann = T.functional.center_crop(fixed_ann.permute(2,0,1), crop_shape).permute(1,2,0)
    # crop_ann = fixed_ann
    # crop_ann = morph.remove_small_objects(crop_ann, min_size=30)

    x_map = torch.zeros(orig_ann.shape[:2], dtype=torch.float32)
    y_map = torch.zeros(orig_ann.shape[:2], dtype=torch.float32)

    inst_list = list(torch.unique(crop_ann))
    inst_list.remove(0)  # 0 is background
    for inst_id in inst_list:
        inst_map = (fixed_ann == inst_id).type(torch.uint8)
        inst_box = get_bounding_box(inst_map).type(torch.uint8)

        # expand the box by 2px
        # Because we first pad the ann at line 207, the bboxes
        # will remain valid after expansion
        inst_box[0] -= 2
        inst_box[2] -= 2
        inst_box[1] += 2
        inst_box[3] += 2

        inst_map = inst_map[inst_box[0] : inst_box[1], inst_box[2] : inst_box[3]]

        if inst_map.shape[0] < 2 or inst_map.shape[1] < 2:
            continue

        # instance center of mass, rounded to nearest pixel
        inst_com = list(measurements.center_of_mass(inst_map.to('cpu')).to('cuda:0'))

        inst_com[0] = int(inst_com[0] + 0.5)
        inst_com[1] = int(inst_com[1] + 0.5)

        inst_x_range = torch.arange(1, inst_map.shape[1] + 1)
        inst_y_range = torch.arange(1, inst_map.shape[0] + 1)
        # shifting center of pixels grid to instance center of mass
        inst_x_range -= inst_com[1]
        inst_y_range -= inst_com[0]

        inst_x, inst_y = torch.meshgrid(inst_x_range, inst_y_range)

        # remove coord outside of instance
        inst_x[inst_map == 0] = 0
        inst_y[inst_map == 0] = 0
        inst_x = inst_x.type("torch.float32")
        inst_y = inst_y.type("torch.float32")

        # normalize min into -1 scale
        if torch.min(inst_x) < 0:
            inst_x[inst_x < 0] /= -torch.min(inst_x[inst_x < 0])
        if torch.min(inst_y) < 0:
            inst_y[inst_y < 0] /= -torch.min(inst_y[inst_y < 0])
        # normalize max into +1 scale
        if torch.max(inst_x) > 0:
            inst_x[inst_x > 0] /= torch.max(inst_x[inst_x > 0])
        if torch.max(inst_y) > 0:
            inst_y[inst_y > 0] /= torch.max(inst_y[inst_y > 0])

        ####
        x_map_box = x_map[inst_box[0] : inst_box[1], inst_box[2] : inst_box[3]]
        x_map_box[inst_map > 0] = inst_x[inst_map > 0]

        y_map_box = y_map[inst_box[0] : inst_box[1], inst_box[2] : inst_box[3]]
        y_map_box[inst_map > 0] = inst_y[inst_map > 0]

    hv_map = torch.dstack([x_map, y_map])
    print(hv_map.shape)
    return hv_map.to('cpu').numpy()

    


