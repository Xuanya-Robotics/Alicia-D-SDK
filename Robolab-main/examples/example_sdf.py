"""
Signed Distance Function (SDF)
==========================
This module provides the SDF class to train a neural network model for the signed distance function from URDF/MJCF files and visualize the reconstructed whole body.
"""
import argparse
import os
import time

import numpy as np
import robolab
import torch


def sdf_from_model(args):
    assert args.modelType in ["NN", "BP"], "Invalid model type. Choose either 'NN' or 'BP'."

    # Instantiate SDF
    sdf_instant = robolab.wdf.SDF(args, model_type=args.modelType)
    asset_path = os.path.join(args.assetRoot, args.assetFile)
    sdf_dir = os.path.join(os.path.dirname(asset_path), "sdf")

    if args.modelType == "NN":
        model_name = f'NN_h{args.hiddenDim}_e{args.trainEpochs}'
        sdf_model_path = os.path.join(sdf_dir, 'NN', f'{model_name}.pt')
    elif args.modelType == "BP":
        model_name = f'BP_{args.numFuncs}'
        sdf_model_path = os.path.join(sdf_dir, 'BP', f'{model_name}.pt')

    if not os.path.exists(sdf_model_path) or args.forceTrain:  # train the model
        sdf_instant.train()
    sdf_model = torch.load(sdf_model_path)

    sdf_instant.create_surface_mesh(sdf_model, nbData=128, vis=True, save_mesh_name=model_name)

    # Generate random points and get the SDF and gradients
    start_time = time.time()
    points = np.random.rand(100, 3) * 2.0 - 1.0
    sdf, gradient = sdf_instant.get_sdf_batch(points, sdf_model, base_trans=None, use_derivative=True)
    print('Time cost:', (time.time() - start_time))
    print('sdf:', sdf.shape, 'gradient:', gradient.shape)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:3' if torch.cuda.is_available() else 'cpu', type=str)

    # Training args
    parser.add_argument('--trainEpochs', default=1000, type=int, help="Epochs for NN training")  # Keep for NN
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
    parser.add_argument('--assetName', default="wheel_chair", type=str,
                        help="Name of the asset (e.g., for finding files)")
    parser.add_argument('--assetRoot', default="../assets", type=str, help="Root directory for assets")
    parser.add_argument('--assetFile', default="mjcf/wheel_chair/wheel_chair.xml", type=str,
                        help="Path to asset file (URDF/MJCF)")

    args = parser.parse_args()

    sdf_from_model(args)
