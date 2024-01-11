import numpy as np

import torch
import torchvision


def comb_video(frame1, frame2):
    assert (frame1.shape == frame2.shape)
    dummy1 = frame1
    dummy2 = frame2
    result = np.concatenate((dummy1 * 255.0, dummy2 * 255.0), axis=1).astype(np.uint8)
    return result


def render_n_save_video(phi_matrices, tempr_matrices, output_video="dendrite_phi_tempr.mp4"):
    total, w, h = phi_matrices.shape

    comb_frames = torch.cat((phi_matrices, tempr_matrices), axis=2)
    print("comb.size=", comb_frames.size())

    comb_frames = (comb_frames * 128.0) + 128.0
    # comb_frames = comb_frames*255.0

    comb_frames = torch.unsqueeze(comb_frames, dim=-1)
    comb_frames = comb_frames.repeat([1, 1, 1, 3])
    print("comb.size=", comb_frames.size())

    torchvision.io.write_video(output_video, comb_frames, fps=15)


def get_phi_tempr(all_data, skip_frame=100):
    phi_matrices = None
    tempr_matrices = None

    for i in range(len(all_data)):
        if i % skip_frame == 0:
            phi = all_data[i]['phi']
            tempr = all_data[i]['Tmpr']

            phi1 = phi.unsqueeze(0)
            tempr1 = tempr.unsqueeze(0)
            if phi_matrices is None:
                phi_matrices = phi1
                tempr_matrices = tempr1
            else:
                phi_matrices = torch.cat((phi_matrices, phi1), 0)
                tempr_matrices = torch.cat((tempr_matrices, tempr1), 0)

    print("max phi=", torch.max(phi_matrices), "min phi=", torch.min(phi_matrices))
    print("max temp=", torch.max(tempr_matrices), "min temp=", torch.min(tempr_matrices))

    max_phi = torch.max(phi_matrices)
    max_tempr = torch.max(tempr_matrices)

    phi_matrices /= max_phi
    tempr_matrices /= max_tempr

    print("phi.shape=", phi_matrices.size())
    print("tempr.shape=", tempr_matrices.size())

    assert phi_matrices.shape == tempr_matrices.shape, tempr_matrices.shape

    return phi_matrices, tempr_matrices


if __name__ == '__main__':
    out_dir = "output/"
    fpath = out_dir + "dendrite_data.pkl"
    output_video = "dendrite_phi_tempr.mp4"

    all_data = torch.load(fpath)
    phi_matrices, tempr_matrices = get_phi_tempr(all_data)

    render_n_save_video(phi_matrices, tempr_matrices, out_dir + output_video)
