import h5py
import numpy as np
from hdmf.backends.hdf5.h5_utils import H5DataIO
from pynwb import NWBFile, NWBHDF5IO
from pynwb.file import Subject
from pynwb.ophys import Fluorescence, RoiResponseSeries
from pynwb.behavior import Position, SpatialSeries
from ndx_multichannel_volume import MultiChannelVolumeSeries
from wbfm.utils.nwb.utils_nwb_export import build_optical_channel_objects, _zimmer_microscope_device
from pynwb import TimeSeries
from datetime import datetime
from dateutil.tz import tzlocal
from wbfm.utils.nwb.utils_nwb_export import CustomDataChunkIterator
import dask.array as da
from pathlib import Path
import os
import argparse


def iter_frames(h5_file, n_timepoints, frame_shape):
    for i in range(n_timepoints):
        data = h5_file[str(i)]["frame"]
        yield da.from_array(data, chunks=frame_shape).transpose((1, 2, 3, 0))  # Lazy loading

def dask_stack_volumes(volume_iter):
    """Stack a generator of volumes into a dask array along time."""
    return da.stack(volume_iter, axis=0)  


def convert_harvard_to_nwb(input_path, 
                           output_path,
                           session_description,
                           identifier,
                           device_name,
                           imaging_rate,
                           DEBUG=False):

    # === USER PARAMETERS ===
    experiment_name = input_path.split("/")[-1].split(".")[0]

    with h5py.File(input_path, "r") as f:

        # === Create NWBFile ===
        nwbfile = NWBFile(
            session_description=session_description,
            identifier=identifier,
            session_start_time=datetime.now(tz=tzlocal()),
            experimenter="Harvard Guy "+str(experiment_name),
            lab="Some Harvard Lab",
            institution="Harvard",
        )

        # === Add Subject ===
        nwbfile.subject = Subject(
            subject_id="subject "+experiment_name,
            species="Caenorhabditis elegans",
            description="Tiny worm",
        )

        # === Add Fluorescence Data ===
        ci_int = f["ci_int"][:]  # shape (97, 1331, 12)
        # Unclear if this is just a list of pixels, but it seems to be... so just take the mean
        ci_mean = ci_int.mean(axis=2)  # shape (97, 1331) → average over 12 pixels

        # Transpose to (time, neurons)
        ci_mean = ci_mean.T  # shape (1331, 97)

        flu_ts = TimeSeries(
        name="mean_fluorescence",
        data=ci_mean,
        unit="a.u.",
        rate=imaging_rate,
        description="Mean calcium intensity per neuron, averaged over ROI pixels"
        )
        nwbfile.add_acquisition(flu_ts)

        # === Add 3D Tracking Data ===
        calcium_imaging_module = nwbfile.create_processing_module(
            name='CalciumActivity',
            description='Calcium time series metadata, segmentation, and fluorescence data'
        )
        points = f["points"][:]  # shape (1331, 98, 3)
        position_module = Position(name="NeuronCentroids")

        for neuron_idx in range(points.shape[1]):
            neuron_trace = points[:, neuron_idx, :]  # (1331, 3)
            print(neuron_trace.shape)

            position_module.add_spatial_series(SpatialSeries(
                name=f"neuron_{(neuron_idx+1):03d}",
                data=neuron_trace,
                unit="micrometers",
                reference_frame="Lab frame",
                description=f"3D position of neuron {neuron_idx}",
                timestamps=np.arange(neuron_trace.shape[0])
            ))

        calcium_imaging_module.add(position_module)

        # Tranpose channel to be last
        frame_shape = np.transpose(f["0/frame"].shape, (1,2,3,0))
        # frame_shape = (320, 192, 20, 2) #np.transpose(f["0/frame"].shape, (1,2,3,0))  # (2, 320, 192, 20)
        chunk_shape = (1,) + frame_shape

        nn_keys = []
        for key in f.keys():
            if key.isdigit():
                nn_keys.append(int(key))
        num_frames = np.array(nn_keys).max() + 1    
        series_shape = (num_frames, ) + frame_shape
        
        # Build metadata objects
        grid_spacing = (0.3, 0.3, 1.75)  # Harvard doesn't actually say what their xy resolution is
        device = nwbfile.create_device(name=device_name)
        CalcImagingVolume, _ = build_optical_channel_objects(device, grid_spacing, ['red', 'green'])
        # Add directly to the file to prevent hdmf.build.errors.OrphanContainerBuildError
        nwbfile.add_imaging_plane(CalcImagingVolume)

        imvol_dask = dask_stack_volumes(iter_frames(f,num_frames, frame_shape))
        chunk_video = (1,) + imvol_dask.shape[1:-1] + (1,)
        video_data = H5DataIO(
            data=CustomDataChunkIterator(array=imvol_dask, chunk_shape=chunk_video, display_progress=True),
            compression="gzip"
        )

        nwbfile.add_acquisition(MultiChannelVolumeSeries(
            name="CalciumImageSeries",
            description="Series of calcium imaging data",
            comments="Calcium imaging data from Harvard lab",
            data=video_data,  # data here should be series of indexed masks
            # Elements below can be kept the same as the CalciumImageSeries defined above
            device=device,
            unit="Voxel gray counts",
            scan_line_rate=2995.,
            dimension=series_shape,  
            resolution=1.,
            # smallest meaningful difference (in specified unit) between values in data: i.e. level of precision
            rate=imaging_rate,  # sampling rate in hz
            imaging_volume=CalcImagingVolume,
        ))


        # === Write NWB file ===
        with NWBHDF5IO(output_path, "w") as io:
            io.write(nwbfile)

    print(f"NWB file written to {output_path}")


if __name__ == "__main__":


    parser = argparse.ArgumentParser(description="Convert Harvard data to NWB format.")
    parser.add_argument('--input_path', type=str, required=True, help='Base directory containing input data as .h5')
    parser.add_argument('--output_path', type=str, required=False, help='Output NWB file path')
    parser.add_argument('--session_description', type=str, default='Harvard Lab Data', help='Session description')
    parser.add_argument('--identifier', type=str, default='samuel_001', help='NWB file identifier')
    parser.add_argument('--device_name', type=str, default='HarvardMicroscope', help='Device name')
    parser.add_argument('--imaging_rate', type=float, default=10.0, help='Imaging rate (Hz)')
    parser.add_argument('--debug', action='store_true', help='If set, only convert the first 10 time points')

    args = parser.parse_args()

    # If the output path is not an absolute path, make it absolute by joining with the base_dir
    if args.output_path is None:
        args.output_path = Path(args.output_path).with_suffix('.nwb')
    if not os.path.isabs(args.output_path):
        args.output_path = os.path.join(args.base_dir, args.output_path)

    convert_harvard_to_nwb(
        base_dir=args.base_dir,
        output_path=args.output_path,
        session_description=args.session_description,
        identifier=args.identifier,
        device_name=args.device_name,
        imaging_rate=args.imaging_rate,
        DEBUG=args.debug
    )

