import urllib.request  # soundata bug workaround: import urllib submodule before soundata loads it
import soundata

DATA_HOME = "data/urbansound8k"


def main():
    dataset = soundata.initialize("urbansound8k", data_home=DATA_HOME)
    dataset.download()
    dataset.validate()


if __name__ == "__main__":
    main()
