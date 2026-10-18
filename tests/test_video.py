"""Episode video: side-by-side camera frames, skipped quietly when imageio/ffmpeg is missing."""

import importlib.util

import numpy as np

from embodied_harness.runtime.trace.video import EpisodeVideo, episode_video_path


def test_each_resume_gets_its_own_episode_video(tmp_path):
    assert episode_video_path(tmp_path) == tmp_path / "episode.mp4"
    assert episode_video_path(tmp_path, 2) == tmp_path / "episode_resume_2.mp4"


def test_four_camera_video_uses_a_two_by_two_grid_without_squashing():
    frames = [np.full((8, 12, 3), value, np.uint8) for value in range(4)]
    composed = EpisodeVideo._compose(frames)
    assert composed is not None and composed.shape == (16, 24, 3)
    assert composed[0, 0, 0] == 0
    assert composed[0, 12, 0] == 1
    assert composed[8, 0, 0] == 2
    assert composed[8, 12, 0] == 3


def test_episode_video_writes_or_skips(tmp_path):
    video = EpisodeVideo(tmp_path / "ep" / "episode.mp4", fps=10)
    for value in range(5):
        video.add(
            {"front": np.full((16, 16, 3), value * 40, np.uint8), "wrist": np.zeros((16, 16, 3), np.uint8)}
        )
    video.add(
        {"front": np.zeros((8, 8, 3), np.uint8), "wrist": np.zeros((16, 16, 3), np.uint8)}
    )  # mismatched: skipped
    video.close()
    has_ffmpeg = (
        importlib.util.find_spec("imageio") is not None
        and importlib.util.find_spec("imageio_ffmpeg") is not None
    )
    if has_ffmpeg:
        assert video.frames == 5 and video.path.stat().st_size > 0
    else:
        assert video.frames == 0 and not video.path.exists()
