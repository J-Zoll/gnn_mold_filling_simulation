from pathlib import Path
import config


def setup_directories():
    Path(config.DIR_DATA_RAW).mkdir(parents=True, exist_ok=True)
    Path(config.DIR_DATA_PROCESSED).mkdir(parents=True, exist_ok=True)
    Path(config.DIR_DATA_STUDIES).mkdir(parents=True, exist_ok=True)
    Path(config.DIR_DATA_MOLDFLOW_OUT).mkdir(parents=True, exist_ok=True)


def main():
    setup_directories()


if __name__ == "__main__":
    main()
