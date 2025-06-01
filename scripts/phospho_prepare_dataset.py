from loguru import logger
from phosphobot.am.base import resize_dataset
from phosphobot.am.gr00t import generate_modality_json
from phosphobot.models.dataset import InfoModel, Dataset
import tyro
from dataclasses import dataclass
from pathlib import Path
from tqdm import tqdm


@dataclass
class Config:
    data_dir: str


from multiprocessing import Pool
from pathlib import Path
import logging
import av
from typing import Tuple
# Assuming InfoModel is imported from your codebase
# from your_module import InfoModel


def resize_video(args: Tuple[Path, Path, Tuple[int, int]]) -> None:
    """Resize a single video file."""
    input_path, output_path, resize_to = args
    try:
        # Open input video
        input_container = av.open(str(input_path))
        input_stream = input_container.streams.video[0]

        # Open output video
        output_container = av.open(str(output_path), mode="w")
        output_stream = output_container.add_stream(
            codec_name="h264",
            rate=input_stream.base_rate,
        )
        output_stream.width = resize_to[0]
        output_stream.height = resize_to[1]
        output_stream.pix_fmt = input_stream.pix_fmt

        # Process frames
        for frame in input_container.decode(video=0):
            frame = frame.reformat(
                width=resize_to[0],
                height=resize_to[1],
            )
            packet = output_stream.encode(frame)
            output_container.mux(packet)

        # Flush encoder
        for packet in output_stream.encode(None):
            output_container.mux(packet)

        input_container.close()
        output_container.close()
    except Exception as e:
        logger.error(f"Error resizing {input_path}: {e}")


def resize_dataset(
    dataset_root_path: Path,
    resize_to: Tuple[int, int] = (320, 240),
) -> Tuple[bool, bool]:
    """
    Resize the dataset to a smaller size for faster training using multi-processing.

    Args:
        dataset_root_path (Path): Path to the dataset root directory.
        resize_to (Tuple[int, int]): Target resolution (width, height).

    Returns:
        Tuple[bool, bool]: (Success flag, Need to recompute stats flag).
    """
    logger.info(f"Resizing videos in {dataset_root_path} to {resize_to[0]}x{resize_to[1]}")
    try:
        meta_path = dataset_root_path / "meta"
        video_information = {}
        validated_info_model = InfoModel.from_json(meta_folder_path=str(meta_path.resolve()))
        for feature in validated_info_model.features.observation_images:
            shape = validated_info_model.features.observation_images[feature].shape
            if shape != [resize_to[1], resize_to[0], 3]:
                video_information[feature] = {
                    "need_to_resize": True,
                    "shape": shape,
                }
                validated_info_model.features.observation_images[feature].shape = [
                    resize_to[1],
                    resize_to[0],
                    3,
                ]
            else:
                logger.info(f"Video {feature} is already in the correct size {shape}")

        if not video_information:
            logger.info("No videos need to be resized.")
            return True, False

        # Collect video paths and prepare tasks
        tasks = []
        for video_folder in video_information:
            if video_information[video_folder]["need_to_resize"]:
                video_path = dataset_root_path / "videos" / "chunk-000" / video_folder
                for episode in video_path.iterdir():
                    if episode.suffix == ".mp4" and not episode.name.startswith("edited_"):
                        out_path = episode.parent / f"edited_{episode.name}"
                        tasks.append((episode, out_path, resize_to))

        # Process videos in parallel using all 22 CPU cores
        # with Pool(processes=22) as pool:
        #     pool.map(resize_video, tasks)
        with Pool(processes=22) as pool:
            for _ in tqdm(pool.imap_unordered(resize_video, tasks), total=len(tasks)):
                pass

        # Rename files after resizing
        for video_folder in video_information:
            if video_information[video_folder]["need_to_resize"]:
                video_path = dataset_root_path / "videos" / "chunk-000" / video_folder
                for episode in video_path.iterdir():
                    if episode.suffix == ".mp4" and episode.name.startswith("edited_"):
                        new_name = episode.name.replace("edited_", "")
                        new_path = episode.parent / new_name
                        new_path.unlink(missing_ok=True)
                        episode.rename(new_path)

        # Save updated info.json
        validated_info_model.to_json(meta_folder_path=str(meta_path.resolve()))

        logger.info("Resizing completed.")
        logger.warning("You now need to recompute the stats for the dataset.")
        return True, True

    except Exception as e:
        logger.error(f"Error resizing videos: {e}")
        return False, False


config = tyro.cli(Config)

# success, recompute_stats = resize_dataset(Path(config.data_dir), resize_to=(224, 224))
# logger.info(f"Resized dataset: {success}")

# logger.info("Generating modality.json file")
# number_of_robots, number_of_cameras = generate_modality_json(Path(config.data_dir))

# Shuffle the dataset
# dataset = Dataset(config.data_dir, enforce_path=False)
# dataset.shuffle_dataset("shuffled_dataset")

# MAke a train and valid split
# shuffled_dataset = Dataset(Path(config.data_dir) / "shuffled_dataset", enforce_path=False)
# split_ratio = 0.9
# first_split_name = "train"
# second_split_name = "valid"
# train_dataset, valid_dataset = shuffled_dataset.split_dataset(split_ratio, first_split_name, second_split_name)

generate_modality_json(Path(config.data_dir))
