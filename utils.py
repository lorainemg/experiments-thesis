from pathlib import Path


def get_plot_folder(folder_path: str):
    """Creates a folder for plots, creating also parents directories if necessary"""
    folder = Path(folder_path)
    if not folder.exists():
        folder.mkdir(parents=True)
    return folder
