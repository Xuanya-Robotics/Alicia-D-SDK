"""
Robot distance field (RDF) with Neural Network
==============================================

This example demonstrates how to use the RDF_NN class to train a Neural Network model for the robot distance field
from URDF/MJCF files and visualize the reconstructed whole body.
"""

import argparse
import os
import time

import robolab
import torch
from robolab.wdf.rdf import RDF  # Changed from RDF


def rdf_from_robot_model(args):
    assert args.modelType in ["NN", "BP"], "Invalid model type. Choose either 'NN' or 'BP'."

    # Instantiate RDF_NN
    rdf_instant = RDF(args, model_type=args.modelType, robot_verbose=True)
    asset_path = os.path.join(args.assetRoot, args.assetFile)
    rdf_dir = os.path.join(os.path.dirname(asset_path), "rdf")

    if args.modelType == "NN":
        model_name = f'NN_h{args.hiddenDim}_e{args.trainEpochs}'
        rdf_model_path = os.path.join(rdf_dir, 'NN', f'{model_name}.pt')
    elif args.modelType == "BP":
        model_name = f'BP_{args.numFuncs}'
        rdf_model_path = os.path.join(rdf_dir, 'BP', f'{model_name}.pt')

    if not os.path.exists(rdf_model_path) or args.forceTrain:  # train the model
        rdf_instant.train()
    rdf_model = torch.load(rdf_model_path)

    rdf_instant.create_surface_mesh(rdf_model, nbData=128, vis=False, save_mesh_name=model_name)

    num_joint = rdf_instant.robot.num_joint
    joint_value = torch.zeros(num_joint).to(args.device)
    base_trans = torch.tensor([[1, 0, 0, 0],
                               [0, 1, 0, 0],
                               [0, 0, 1, 0],
                               [0, 0, 0, 1]]).float().to(args.device)
    trans_dict = rdf_instant.robot.get_trans_dict(joint_value, base_trans)
    # visualize the Bernstein Polynomial model for the whole body
    # rdf.visualize_reconstructed_whole_body(rdf_model, trans_dict, tag=model_name)

    # Run RDF_NN inference
    batch_size = 1024
    num_points = 64
    x = torch.rand(batch_size, num_points, 3).to(args.device) * 2.0 - 1.0  # [B, N, 3]
    joint_value = torch.rand(batch_size, rdf_instant.robot.num_joint).to(args.device).float()  # [B, num_joint]
    base_trans = torch.eye(4, device=args.device).unsqueeze(0).expand(batch_size, 4, 4)  # [B, 4, 4]

    start_time = time.time()
    sdf, gradient = rdf_instant.get_whole_body_sdf_batch(x, joint_value, rdf_model, base_trans=base_trans,
                                                         use_derivative=True)
    print('Time cost:', (time.time() - start_time))
    print('sdf:', sdf.shape, 'gradient:', gradient.shape)

    # joint_value = torch.zeros(num_joint).to(args.device).reshape((-1, num_joint))
    joint_value = torch.rand(num_joint).to(args.device).reshape((-1, num_joint))
    robolab.wdf.plot_3D_sdf_with_gradient(joint_value, rdf_instant, model=rdf_model, device=args.device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:3' if torch.cuda.is_available() else 'cpu', type=str)

    # Training args
    parser.add_argument('--trainEpochs', default=300, type=int, help="Epochs for NN training")  # Keep for NN
    parser.add_argument('--forceTrain', action='store_true', help="Force training even if model exists")
    parser.add_argument('--saveMeshDict', action='store_false', help="Save trained NN models")  # Keep general name
    parser.add_argument('--samplePoints', action='store_true', help="Force resampling points (if not existing)")
    parser.add_argument('--parallel', action='store_false', help="Use multiprocessing for point sampling")
    parser.add_argument('--domainMax', default=1.0, type=float)
    parser.add_argument('--domainMin', default=-1.0, type=float)
    parser.add_argument('--modelType', default="NN", type=str)  # BP or NN
    # NN specific args
    parser.add_argument('--hiddenDim', default=256, type=int, help="Hidden dimension size for SDF NN")
    parser.add_argument('--learningRate', default=1e-4, type=float, help="Learning rate for NN training")
    parser.add_argument('--nnBatchSize', default=8192000, type=int, help="Batch size for NN training")
    # BP specific args
    parser.add_argument('--numFuncs', default=16, type=int)

    # Asset args
    parser.add_argument('--assetName', default="Bruce", type=str, help="Name of the asset (e.g., for finding files)")
    parser.add_argument('--assetRoot', default="../assets", type=str, help="Root directory for assets")
    parser.add_argument('--assetFile', default="mjcf/bruce/bruce.xml", type=str, help="Path to asset file (URDF/MJCF)")
    parser.add_argument('--baseLink', default="pelvis", type=str, help="Base link of the robot")

    args = parser.parse_args()

    rdf_from_robot_model(args)
